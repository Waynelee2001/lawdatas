from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Generator, Iterable

from llama_index.core import VectorStoreIndex

from rag.loader import build_article_lookup
from rag.query_analysis import QueryProfile, analyze_query
from rag.schema import LegalArticle
from rag.siliconflow_rerank import SiliconFlowReranker

try:
    import jieba
    import jieba.analyse
    from rank_bm25 import BM25Okapi

    jieba.setLogLevel(logging.WARNING)
    _BM25_AVAILABLE = True
except ImportError:
    jieba = None  # type: ignore[assignment]
    BM25Okapi = None  # type: ignore[assignment]
    _BM25_AVAILABLE = False

logger = logging.getLogger(__name__)

QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}")
GENERIC_RULE_HINTS = ("基本", "原则", "规则", "一般", "总则")
LISTING_QUERY_HINTS = (
    "哪些",
    "有哪些",
    "情形",
    "规定",
    "司法解释",
    "解释",
    "适用",
    "无效",
)
JUDICIAL_INTERPRETATION_HINTS = (
    "最高人民法院",
    "解释",
    "规定",
    "意见",
    "批复",
    "若干问题",
)

# Common Chinese stop words that add no retrieval signal
_BM25_STOP_WORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "和",
        "是",
        "在",
        "有",
        "与",
        "对",
        "也",
        "其",
        "中",
        "为",
        "以",
        "或",
        "及",
        "等",
        "被",
        "将",
        "由",
        "按",
        "向",
        "该",
        "此",
        "本",
        "其他",
        "相关",
        "如果",
        "如",
        "则",
        "但",
        "应当",
        "应",
        "可以",
        "不得",
        "不",
        "可",
        "之",
        "于",
        "时",
        "当",
        "即",
        "并",
        "而",
        "且",
        "又",
        "亦",
        "均",
        "各",
        "共",
    }
)

# RRF constant — 60 is the standard default from the original paper
_RRF_K = 60


@dataclass
class RetrievedArticle:
    article: LegalArticle
    score: float
    semantic_rerank_score: float = 0.0
    rerank_score: float = 0.0
    reasons: set[str] = field(default_factory=set)
    matched_terms: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BM25 sparse index
# ---------------------------------------------------------------------------


# Matches full article-number tokens like 第三百一十一条 or 第二十五条之一
_ARTICLE_NUM_RE = re.compile(
    r"第[一二三四五六七八九十百千万零〇○0-9]+"
    r"(?:百[一二三四五六七八九十]*)?"
    r"(?:[一二三四五六七八九十]+)?"
    r"条(?:之[一二三四五六七八九十百千万零〇○0-9]+)?"
)


def _split_preserving_article_nums(text: str) -> Generator[str, None, None]:
    """
    Split *text* into segments, yielding article-number tokens intact so that
    jieba cannot break them into sub-tokens.  Non-article-number segments are
    yielded as plain strings for jieba to process normally.
    """
    pos = 0
    for m in _ARTICLE_NUM_RE.finditer(text):
        start, end = m.start(), m.end()
        if start > pos:
            yield text[pos:start]  # regular text before the match
        yield "\x00" + m.group()  # sentinel-prefixed article token
        pos = end
    if pos < len(text):
        yield text[pos:]


def _bm25_tokenize(text: str) -> list[str]:
    """
    Tokenize Chinese legal text for BM25 using jieba word segmentation.

    Article-number tokens (e.g. "第三百一十一条") are preserved intact before
    jieba runs so they are never split into meaningless sub-tokens.
    Filters single-character tokens (mostly particles) and stop words.
    """
    tokens: list[str] = []

    if not _BM25_AVAILABLE:
        # Fallback: simple word split when jieba is not installed
        for m in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", text):
            m = m.strip()
            if m and m not in _BM25_STOP_WORDS:
                tokens.append(m)
        return tokens

    for segment in _split_preserving_article_nums(text):
        if segment.startswith("\x00"):
            # This is a pre-served article-number token — keep as-is
            tokens.append(segment[1:])
            continue
        cut_fn = jieba.cut if jieba is not None else None  # type: ignore[union-attr]
        if cut_fn is None:
            tokens.extend(
                m
                for m in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", segment)
                if m not in _BM25_STOP_WORDS
            )
            continue
        for token in cut_fn(segment):
            token = token.strip()
            if not token:
                continue
            if token in _BM25_STOP_WORDS:
                continue
            # Drop isolated single Chinese characters (mostly particles)
            if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
                continue
            tokens.append(token)
    return tokens


def _article_bm25_text(article: LegalArticle) -> str:
    """Concatenated text used when building the BM25 index for an article."""
    parts: list[str] = [article.law_name, article.article_num]
    if article.annotation:
        parts.append(article.annotation)
    if article.chapter:
        parts.append(article.chapter)
    if article.chapter_annotation:
        parts.append(article.chapter_annotation)
    # Index the first 600 characters of the article body — enough for keyword
    # signal without bloating the index with boilerplate tail text.
    parts.append(article.article_text[:600])
    return " ".join(parts)


class BM25ArticleIndex:
    """
    Sparse keyword index over the legal article corpus.

    Build this once per corpus and cache it at the service layer.  It uses
    jieba for Chinese word segmentation and rank_bm25's BM25Okapi scoring.
    """

    def __init__(self, articles: list[LegalArticle]) -> None:
        self.articles = articles
        self._business_id_to_idx: dict[str, int] = {
            a.business_id: i for i, a in enumerate(articles)
        }
        logger.info("Building BM25 index for %d articles …", len(articles))
        corpus = [_bm25_tokenize(_article_bm25_text(a)) for a in articles]
        if not _BM25_AVAILABLE or BM25Okapi is None:
            raise RuntimeError(
                "rank_bm25 and jieba are required for BM25ArticleIndex. "
                "Run: pip install rank-bm25 jieba"
            )
        self._bm25 = BM25Okapi(corpus)  # type: ignore[misc]
        logger.info("BM25 index ready (%d articles).", len(articles))

    def search(self, query: str, top_k: int = 20) -> list[tuple[LegalArticle, float]]:
        """Return up to *top_k* (article, bm25_score) pairs, best first."""
        tokens = _bm25_tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # argsort gives ascending order; reverse-slice to get top scores
        sorted_indices = scores.argsort()[::-1]
        results: list[tuple[LegalArticle, float]] = []
        for idx in sorted_indices[:top_k]:
            score = float(scores[idx])
            if score <= 0.0:
                break
            results.append((self.articles[int(idx)], score))
        return results


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def _apply_rrf_from_bm25(
    candidates: dict[str, RetrievedArticle],
    bm25_hits: list[tuple[LegalArticle, float]],
    article_lookup: dict[str, LegalArticle],
) -> None:
    """
    Merge BM25 sparse-retrieval results into the candidate pool via
    Reciprocal Rank Fusion (RRF).

    - Articles already present (found by vector search) receive an additive
      RRF contribution and the "bm25" reason tag.
    - Articles only found by BM25 are added as new candidates.
    """
    for rank, (article, _bm25_score) in enumerate(bm25_hits):
        rrf_contribution = 1.0 / (_RRF_K + rank + 1)
        existing = candidates.get(article.business_id)
        if existing is None:
            # Prefer the fully-populated article from the lookup if available
            full_article = article_lookup.get(article.business_id, article)
            candidates[article.business_id] = RetrievedArticle(
                article=full_article,
                score=rrf_contribution,
                reasons={"bm25"},
            )
        else:
            # Both retrieval paths agree: stronger evidence → additive score
            existing.score = existing.score + rrf_contribution
            existing.reasons.add("bm25")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_result_metadata(result: object) -> tuple[dict, str]:
    if hasattr(result, "node"):
        node = getattr(result, "node")
        return getattr(node, "metadata", {}), getattr(node, "text", "")
    return getattr(result, "metadata", {}), getattr(result, "text", "")


def _score_value(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _add_candidate(
    candidates: dict[str, RetrievedArticle],
    article: LegalArticle,
    score: float,
    reason: str,
) -> None:
    existing = candidates.get(article.business_id)
    if existing is None:
        candidates[article.business_id] = RetrievedArticle(
            article=article,
            score=score,
            reasons={reason},
        )
        return
    existing.score = max(existing.score, score)
    existing.reasons.add(reason)


def _expand_business_ids(article: LegalArticle) -> Iterable[tuple[str, str]]:
    for edge in article.outgoing_citations:
        if edge.target_business_id:
            yield edge.target_business_id, "outgoing_citation"
    for edge in article.incoming_citations:
        if edge.source_business_id:
            yield edge.source_business_id, "incoming_citation"


def _extract_query_terms(query: str) -> list[str]:
    normalized = query
    for token in (
        "中",
        "关于",
        "对于",
        "什么",
        "哪些",
        "如何",
        "怎么",
        "规定",
        "基本规则",
        "基本原则",
    ):
        normalized = normalized.replace(token, " ")
    seen: set[str] = set()
    terms: list[str] = []
    for match in QUERY_TOKEN_RE.findall(normalized):
        token = match.strip()
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _find_anchor_articles(
    article_lookup: dict[str, LegalArticle], query_profile: QueryProfile
) -> list[LegalArticle]:
    anchors: list[LegalArticle] = []
    seen: set[str] = set()
    if not query_profile.anchor_refs:
        return anchors
    for article in article_lookup.values():
        for anchor_ref in query_profile.anchor_refs:
            if (
                article.law_name == anchor_ref.law_name
                and article.article_num == anchor_ref.article_num
            ):
                if article.business_id not in seen:
                    anchors.append(article)
                    seen.add(article.business_id)
    return anchors


def _wants_linked_authorities(query: str) -> bool:
    return any(hint in query for hint in LISTING_QUERY_HINTS)


def _compute_rerank(
    result: RetrievedArticle,
    query: str,
    query_terms: list[str],
    query_profile: QueryProfile,
) -> None:
    article = result.article
    rerank_score = result.semantic_rerank_score or result.score
    matched_terms: list[str] = []
    wants_linked_authorities = _wants_linked_authorities(query)
    search_text = " ".join(
        [
            article.law_name,
            article.article_num,
            article.annotation,
            article.chapter,
            article.chapter_annotation,
            article.article_text[:300],
        ]
    )

    # --- exact references in the user query ---------------------------------
    if article.law_name in query:
        rerank_score += 0.2
        matched_terms.append(article.law_name)
    if article.article_num in query:
        rerank_score += 0.25
        matched_terms.append(article.article_num)

    # --- retrieval source bonuses -------------------------------------------
    if "vector" in result.reasons:
        rerank_score += 0.12
    if "bm25" in result.reasons:
        # Keyword overlap confirmed by sparse index
        rerank_score += 0.10
        matched_terms.append("bm25_hit")
    if "vector" in result.reasons and "bm25" in result.reasons:
        # Both retrieval systems agree: very strong signal
        rerank_score += 0.08
        matched_terms.append("hybrid_hit")
    if len(result.reasons) > 1:
        rerank_score += 0.04
    if "incoming_citation" in result.reasons:
        # Only award the citation bonus when the article's own content has
        # some term overlap with the query.  A pure structural hit (引用关系
        # 带进来但内容与问题无关) should not get a free score boost.
        citation_overlap = (
            any(term in search_text for term in query_terms) if query_terms else False
        )
        if citation_overlap:
            rerank_score += 0.08
            matched_terms.append("incoming_citation")
        else:
            # Penalise: pulled in only by graph, content doesn't match query
            rerank_score -= 0.06
            matched_terms.append("incoming_citation_no_overlap")
    if "outgoing_citation" in result.reasons:
        rerank_score += 0.05
        matched_terms.append("outgoing_citation")

    # --- term-level matching ------------------------------------------------
    for term in query_terms:
        if term in search_text:
            rerank_score += 0.03
            matched_terms.append(term)
            if term in article.annotation:
                rerank_score += 0.05
            if term in article.law_name:
                rerank_score += 0.04

    # --- LLM scenario terms (from expanded QueryProfile) -------------------
    if query_profile.scenario_terms:
        for term in query_profile.scenario_terms:
            if term and term in search_text:
                rerank_score += 0.04
                matched_terms.append(f"scenario:{term}")
                if term in article.annotation:
                    rerank_score += 0.03

    # --- generic "rules / principles" queries -------------------------------
    if any(hint in query for hint in GENERIC_RULE_HINTS):
        if article.chapter.startswith("第一章") or "总则" in article.chapter_annotation:
            rerank_score += 0.08
        if any(hint in article.annotation for hint in ("基本", "原则", "总则")):
            rerank_score += 0.08

    # --- listing queries (e.g. "哪些情形") ----------------------------------
    if wants_linked_authorities and "incoming_citation" in result.reasons:
        rerank_score += 0.22
        if article.law_name.startswith("最高人民法院"):
            rerank_score += 0.18
            matched_terms.append("司法解释")
        if any(hint in article.law_name for hint in JUDICIAL_INTERPRETATION_HINTS):
            rerank_score += 0.1
        if any(
            token in article.annotation for token in ("无效", "效力", "强制", "合同")
        ):
            rerank_score += 0.08
        if any(
            token in article.article_text[:200]
            for token in ("无效", "强制性规定", "效力")
        ):
            rerank_score += 0.06

    # --- hardcoded topic: 股权善意取得 (preserved from original) -----------
    if query_profile.topic == "股权善意取得":
        if any(token in search_text for token in query_profile.scenario_terms):
            rerank_score += 0.12
            matched_terms.append("topic_scenario")
        if any(
            token in article.annotation
            for token in ("股权", "股东", "名义股东", "处分股权", "出资")
        ):
            rerank_score += 0.14
        if article.law_name.startswith(
            "最高人民法院关于适用《中华人民共和国公司法》若干问题的规定（三）"
        ):
            rerank_score += 0.18
            matched_terms.append("公司法规定（三）")
        if article.article_num in ("第二十五条", "第二十七条", "第七条"):
            rerank_score += 0.22
            matched_terms.append("股权善意取得核心条文")
        if "破产法" in article.law_name:
            rerank_score -= 0.25
        if any(
            token in article.annotation
            for token in (
                "善意取得",
                "名义股东处分股权的效力",
                "股权转让后原股东处分股权",
                "以无处分权财产出资的效力",
            )
        ):
            rerank_score += 0.18

    # --- preferred / suppressed law keywords from QueryProfile --------------
    if query_profile.preferred_law_keywords and any(
        token in article.law_name for token in query_profile.preferred_law_keywords
    ):
        rerank_score += 0.06
    if query_profile.suppress_law_keywords and any(
        token in article.law_name for token in query_profile.suppress_law_keywords
    ):
        rerank_score -= 0.14

    result.rerank_score = rerank_score
    result.matched_terms = sorted(set(matched_terms))


def _build_rerank_document(article: LegalArticle) -> str:
    lines = [
        f"法律名称：{article.law_name}",
        f"条号：{article.article_num}",
    ]
    if article.annotation:
        lines.append(f"标注：{article.annotation}")
    lines.append(f"正文：{article.article_text[:600]}")
    return "\n".join(lines)


def _apply_semantic_rerank(
    reranker: SiliconFlowReranker | None,
    query: str,
    candidates: list[RetrievedArticle],
) -> None:
    if reranker is None or not candidates:
        return
    documents = [_build_rerank_document(c.article) for c in candidates]
    results = reranker.rerank(query=query, documents=documents, top_n=len(documents))
    for item in results:
        if 0 <= item.index < len(candidates):
            candidates[item.index].semantic_rerank_score = item.relevance_score


# ---------------------------------------------------------------------------
# Public retrieval entry point
# ---------------------------------------------------------------------------


def graph_retrieve(
    index: VectorStoreIndex,
    query: str,
    *,
    top_k: int = 5,
    graph_expand_k: int = 2,
    law_ids: set[str] | None = None,
    reranker: SiliconFlowReranker | None = None,
    bm25_index: BM25ArticleIndex | None = None,
    query_profile: QueryProfile | None = None,
) -> list[RetrievedArticle]:
    """
    Hybrid, graph-aware retrieval pipeline.

    Steps
    -----
    1. Query analysis — use the supplied *query_profile* (from LLM) or fall
       back to rule-based analysis.
    2. Dense vector retrieval via LlamaIndex + Qdrant.
    3. Sparse BM25 retrieval (when *bm25_index* is provided), merged with the
       vector candidates using Reciprocal Rank Fusion.
    4. Anchor article injection for queries that reference a specific article.
    5. One-hop citation graph expansion from all seed candidates.
    6. Optional semantic reranking via the SiliconFlow reranker model.
    7. Rule-based reranking that applies query-profile awareness (preferred
       laws, scenario terms, judicial-interpretation bonuses, etc.).

    Parameters
    ----------
    index:
        Pre-built LlamaIndex VectorStoreIndex.
    query:
        Natural-language legal question.
    top_k:
        Number of final results to return.
    graph_expand_k:
        Maximum citation neighbours to expand per seed article.
    law_ids:
        If provided, restrict retrieval to articles from these law IDs.
    reranker:
        Optional SiliconFlow semantic reranker.
    bm25_index:
        Optional pre-built BM25ArticleIndex for hybrid retrieval.
    query_profile:
        Pre-computed QueryProfile (e.g. from LLM analysis in service.py).
        If None, rule-based analyze_query() is used.
    """

    # ------------------------------------------------------------------ 1 --
    if query_profile is None:
        query_profile = analyze_query(query)

    article_lookup = build_article_lookup(law_ids=law_ids)

    # ------------------------------------------------------------------ 2 --
    # Dense vector retrieval — use a wider candidate window (2×) so downstream
    # stages have more material to work with.
    retriever = index.as_retriever(similarity_top_k=top_k * 2)
    raw_results = retriever.retrieve(query_profile.expanded_query)

    candidates: dict[str, RetrievedArticle] = {}
    seed_articles: list[RetrievedArticle] = []

    for result in raw_results:
        metadata, _ = _get_result_metadata(result)
        business_id = metadata.get("business_id")
        if not business_id:
            continue
        article = article_lookup.get(str(business_id))
        if article is None:
            continue
        seed = RetrievedArticle(
            article=article,
            score=_score_value(getattr(result, "score", 0.0)),
            reasons={"vector"},
        )
        seed_articles.append(seed)
        candidates[business_id] = seed

    # ------------------------------------------------------------------ 3 --
    if bm25_index is not None:
        # BM25 uses the expanded query so that LLM-added terms improve recall
        bm25_query = query_profile.expanded_query
        bm25_hits = bm25_index.search(bm25_query, top_k=top_k * 3)

        if law_ids is not None:
            bm25_hits = [(art, sc) for art, sc in bm25_hits if art.law_id in law_ids]

        _apply_rrf_from_bm25(candidates, bm25_hits, article_lookup)

        # BM25-only hits become additional seeds for graph expansion
        for article, _ in bm25_hits[:top_k]:
            biz_id = article.business_id
            if biz_id in candidates and "vector" not in candidates[biz_id].reasons:
                seed_articles.append(candidates[biz_id])

    # ------------------------------------------------------------------ 4 --
    for anchor_article in _find_anchor_articles(article_lookup, query_profile):
        existing = candidates.get(anchor_article.business_id)
        if existing is None:
            seed = RetrievedArticle(
                article=anchor_article,
                score=0.75,
                reasons={"anchor_article"},
            )
            seed_articles.append(seed)
            candidates[anchor_article.business_id] = seed
        else:
            existing.score = max(existing.score, 0.75)
            existing.reasons.add("anchor_article")

    # ------------------------------------------------------------------ 5 --
    for seed in seed_articles:
        expanded = 0
        for business_id, reason in _expand_business_ids(seed.article):
            article = article_lookup.get(business_id)
            if article is None:
                continue
            expanded += 1
            graph_score = seed.score * (0.85 if reason == "outgoing_citation" else 0.8)
            _add_candidate(candidates, article, graph_score, reason)
            if expanded >= graph_expand_k:
                break

    # ------------------------------------------------------------------ 6 --
    candidate_list = list(candidates.values())
    _apply_semantic_rerank(reranker, query, candidate_list)

    # ------------------------------------------------------------------ 7 --
    query_terms = _extract_query_terms(query_profile.expanded_query)
    for candidate in candidate_list:
        _compute_rerank(candidate, query, query_terms, query_profile)

    return sorted(
        candidate_list,
        key=lambda item: (
            -item.rerank_score,
            -item.semantic_rerank_score,
            -len(item.reasons),
            -item.score,
            item.article.law_name,
            item.article.article_num,
        ),
    )
