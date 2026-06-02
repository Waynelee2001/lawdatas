from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import requests

from rag.query_analysis import analyze_query
from rag.retrieval import RetrievedArticle
from rag.topic_grouping import group_topic_candidates

LISTING_QUERY_HINTS = ("哪些", "有哪些", "规定", "情形", "规则", "法条", "司法解释")
TOPIC_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}")


@dataclass
class CompressedEvidence:
    primary_basis: list[dict[str, Any]]
    supporting_basis: list[dict[str, Any]]
    exceptions_or_limits: list[str]
    procedural_links: list[str]
    citation_paths: list[str]
    compressed_summary: str
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SiliconFlowChatCompressor:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY is required")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def compress(
        self, *, query: str, candidates: list[RetrievedArticle], top_n: int = 6
    ) -> CompressedEvidence:
        if not candidates:
            return fallback_evidence(query=query, candidates=[])

        chosen = candidates[: max(1, min(top_n, len(candidates)))]
        payload = {
            "model": self.model_name,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": _COMPRESS_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": _build_prompt(query, chosen),
                },
            ],
        }
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        evidence = normalize_compressed_evidence(parsed, chosen)
        # Only supplement when the LLM failed to produce meaningful primary
        # basis — if DeepSeek already returned non-empty primary_basis, trust
        # its relevance filtering and do not override with the mechanical
        # supplement step (which ignores the LLM's exclusion decisions).
        if is_listing_query(query) and not evidence.primary_basis:
            evidence = supplement_listing_evidence(query, evidence, chosen)
        evidence.fallback_used = False
        return evidence


# ---------------------------------------------------------------------------
# Compression prompt
# ---------------------------------------------------------------------------

_COMPRESS_SYSTEM_PROMPT = """\
你是法律RAG证据压缩器，服务于中国法律职业资格考试（法考）场景。

## 核心任务
根据用户问题，从候选法条中筛选出真正相关的条文，整理为结构化证据。

## 相关性判断（最重要）
每条候选法条都必须通过以下相关性门槛，才能进入 primary_basis 或 supporting_basis：

1. **内容直接相关**：法条的核心内容必须直接回答用户问题所涉及的法律问题。
   - 例如，用户问"股权善意取得"，法条必须直接规定股权转让中第三人善意取得股权的规则。

2. **引用不等于相关**：某法条仅因为引用了同一基础条文（如民法典第311条）而被召回，
   但其核心内容是另一个不同的法律问题（如"出资人以无处分权财产出资"、"合同无效"等），
   则该法条不应纳入 primary_basis 或 supporting_basis。
   可以在 citation_paths 中简短提及，说明其与本问题的区别。

3. **区分主题**：请仔细阅读每条法条的正文（text字段），判断它究竟在规范什么法律关系，
   不要仅凭标注（annotation）或引用关系（reasons字段）判断相关性。

## 输出格式
输出必须是合法JSON对象，字段固定如下（全部必填，不相关时填空数组/空字符串）：

{
  "primary_basis": [
    {
      "law_id": "...",
      "law_name": "...",
      "article_num": "...",
      "annotation": "...",
      "reason": "说明该条为何直接回答用户问题（一句话）"
    }
  ],
  "supporting_basis": [
    {
      "law_id": "...",
      "law_name": "...",
      "article_num": "...",
      "annotation": "...",
      "reason": "说明该条对主依据的补充或细化作用（一句话）"
    }
  ],
  "exceptions_or_limits": ["例外情形或限制条件，字符串列表"],
  "procedural_links": ["程序性衔接规定，字符串列表"],
  "citation_paths": ["引用路径说明，包括被排除但引用相关的法条及排除理由"],
  "compressed_summary": "100字以内的中文摘要，只陈述与用户问题直接相关的核心规则"
}

## 其他规则
- 不得编造法条或条号，只能使用候选列表中出现的法条。
- reason 字段必须解释实质关联，不能只写"相关"或"引用"。
- 如果候选中没有真正相关的法条，primary_basis 可以为空数组，并在 compressed_summary 中说明。
- 对于列举型问题（"有哪些规定/情形"），supporting_basis 应尽量覆盖，但每条仍须通过相关性门槛。
"""


def _build_prompt(query: str, candidates: list[RetrievedArticle]) -> str:
    blocks = []
    for idx, item in enumerate(candidates, start=1):
        article = item.article
        retrieval_path = ", ".join(sorted(item.reasons))
        blocks.append(
            "\n".join(
                [
                    f"【候选{idx}】",
                    f"law_id: {article.law_id}",
                    f"law_name: {article.law_name}",
                    f"article_num: {article.article_num}",
                    f"annotation: {article.annotation or '（无标注）'}",
                    f"chapter: {article.chapter or ''}",
                    f"retrieval_path: {retrieval_path}",
                    f"  （retrieval_path 说明：vector=向量语义召回，bm25=关键词召回，"
                    f"incoming_citation=被其他法条引用而扩展，outgoing_citation=引用了其他法条而扩展，"
                    f"anchor_article=查询中明确点名）",
                    f"rerank_score: {item.rerank_score:.3f}",
                    f"text: {article.article_text[:800]}",
                ]
            )
        )

    intro = (
        f"用户问题：{query}\n\n"
        "请严格按照系统提示中的【相关性判断】规则，逐条审查以下候选法条，"
        "判断每条是否真正回答了用户的具体法律问题，然后整理成结构化证据。\n"
        "特别注意：retrieval_path 中含 incoming_citation 的法条，"
        "是通过引用图扩展召回的，请务必阅读其正文确认内容相关性，"
        "不要仅因为它引用了某基础条文就认为它相关。\n\n"
        "候选法条如下：\n\n"
    )
    return intro + "\n\n".join(blocks)


def normalize_compressed_evidence(
    payload: dict[str, Any], candidates: list[RetrievedArticle]
) -> CompressedEvidence:
    candidate_map = {
        (item.article.law_id, item.article.article_num): item for item in candidates
    }

    def norm_basis(items: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if not isinstance(items, list):
            return normalized
        for item in items:
            if not isinstance(item, dict):
                continue
            law_id = str(item.get("law_id", "")).strip()
            article_num = str(item.get("article_num", "")).strip()
            matched = candidate_map.get((law_id, article_num))
            if matched is None:
                continue
            article = matched.article
            normalized.append(
                {
                    "law_id": article.law_id,
                    "law_name": article.law_name,
                    "article_num": article.article_num,
                    "annotation": article.annotation,
                    "reason": item.get("reason") or "候选依据",
                    "score": matched.rerank_score,
                }
            )
        return normalized

    return CompressedEvidence(
        primary_basis=norm_basis(payload.get("primary_basis")),
        supporting_basis=norm_basis(payload.get("supporting_basis")),
        exceptions_or_limits=_norm_str_list(payload.get("exceptions_or_limits")),
        procedural_links=_norm_str_list(payload.get("procedural_links")),
        citation_paths=_norm_str_list(payload.get("citation_paths")),
        compressed_summary=str(payload.get("compressed_summary") or "").strip(),
        fallback_used=False,
    )


def _norm_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def is_listing_query(query: str) -> bool:
    return any(hint in query for hint in LISTING_QUERY_HINTS)


def topic_keywords(query: str) -> list[str]:
    keywords: list[str] = []
    for token in TOPIC_TOKEN_RE.findall(query):
        token = token.strip()
        if len(token) >= 2 and token not in keywords:
            keywords.append(token)
    if "股权" in query:
        for token in ("股权", "股东", "公司", "名义股东", "转让", "处分"):
            if token not in keywords:
                keywords.append(token)
    if "善意取得" in query:
        for token in ("善意", "善意取得", "无处分权"):
            if token not in keywords:
                keywords.append(token)
    return keywords


def candidate_topic_score(item: RetrievedArticle, keywords: list[str]) -> int:
    article = item.article
    search_text = " ".join(
        [
            article.law_name,
            article.annotation,
            article.article_text[:260],
        ]
    )
    score = 0
    for token in keywords:
        if token and token in search_text:
            score += 1
        if token and token in article.annotation:
            score += 2
        if token and token in article.law_name:
            score += 2
    if "incoming_citation" in item.reasons:
        score += 1
    return score


def passes_topic_filter(query: str, item: RetrievedArticle) -> bool:
    article = item.article
    search_text = " ".join(
        [article.law_name, article.annotation, article.article_text[:260]]
    )
    if "股权" in query:
        if not any(
            token in search_text for token in ("股权", "股东", "公司", "名义股东")
        ):
            return False
    return True


def fallback_evidence(
    *, query: str, candidates: list[RetrievedArticle]
) -> CompressedEvidence:
    primary = []
    supporting = []
    citation_paths = []
    for idx, item in enumerate(candidates[:6]):
        target = primary if idx < 2 else supporting
        target.append(
            {
                "law_id": item.article.law_id,
                "law_name": item.article.law_name,
                "article_num": item.article.article_num,
                "annotation": item.article.annotation,
                "reason": ", ".join(sorted(item.reasons)) or "候选依据",
                "score": item.rerank_score,
            }
        )
        if "incoming_citation" in item.reasons:
            citation_paths.append(
                f"{item.article.law_name}{item.article.article_num} <- 反向引用展开"
            )
    summary = "；".join(
        f"{entry['law_name']}{entry['article_num']}[{entry['annotation'] or '无标注'}]"
        for entry in primary[:2]
    )
    if not summary:
        summary = f"未找到与“{query}”直接相关的压缩证据。"
    evidence = CompressedEvidence(
        primary_basis=primary,
        supporting_basis=supporting,
        exceptions_or_limits=[],
        procedural_links=[],
        citation_paths=citation_paths,
        compressed_summary=summary,
        fallback_used=True,
    )
    if is_listing_query(query):
        evidence = supplement_listing_evidence(query, evidence, candidates[:10])
    return evidence


def supplement_listing_evidence(
    query: str,
    evidence: CompressedEvidence,
    candidates: list[RetrievedArticle],
    max_supporting: int = 8,
) -> CompressedEvidence:
    query_profile = analyze_query(query)
    topic_groups = group_topic_candidates(
        query, candidates, limit_per_group=max_supporting
    )
    query_hint_keywords = topic_keywords(query)
    seen = {
        (item["law_id"], item["article_num"])
        for item in evidence.primary_basis + evidence.supporting_basis
    }
    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            -candidate_topic_score(item, query_hint_keywords),
            "incoming_citation" not in item.reasons,
            -item.rerank_score,
        ),
    )
    evidence.supporting_basis = [
        basis
        for basis in evidence.supporting_basis
        if any(
            basis["law_id"] == item.article.law_id
            and basis["article_num"] == item.article.article_num
            and candidate_topic_score(item, query_hint_keywords) > 0
            and passes_topic_filter(query, item)
            for item in candidates
        )
    ]
    seen = {
        (item["law_id"], item["article_num"])
        for item in evidence.primary_basis + evidence.supporting_basis
    }
    for item in ranked_candidates:
        article = item.article
        key = (article.law_id, article.article_num)
        if key in seen:
            continue
        if "incoming_citation" not in item.reasons:
            continue
        if candidate_topic_score(item, query_hint_keywords) <= 0:
            continue
        if not passes_topic_filter(query, item):
            continue
        evidence.supporting_basis.append(
            {
                "law_id": article.law_id,
                "law_name": article.law_name,
                "article_num": article.article_num,
                "annotation": article.annotation,
                "reason": "反向引用补充",
                "score": item.rerank_score,
            }
        )
        seen.add(key)
        if len(evidence.supporting_basis) >= max_supporting:
            break

    if query_profile.topic == "股权善意取得":
        prioritized = topic_groups.get("topic_specific_rules", [])
        supporting_map = {
            (item["law_id"], item["article_num"]): item
            for item in evidence.supporting_basis
        }
        reordered: list[dict[str, Any]] = []
        for item in prioritized:
            key = (item["law_id"], item["article_num"])
            existing = supporting_map.pop(key, None)
            reordered.append(
                existing
                or {
                    "law_id": item["law_id"],
                    "law_name": item["law_name"],
                    "article_num": item["article_num"],
                    "annotation": item["annotation"],
                    "reason": "专题专门规则",
                    "score": item["score"],
                }
            )
        evidence.supporting_basis = reordered + list(supporting_map.values())
        evidence.supporting_basis = evidence.supporting_basis[:max_supporting]
    return evidence
