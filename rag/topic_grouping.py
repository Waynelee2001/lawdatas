from __future__ import annotations

import re
from typing import Any

from rag.query_analysis import QueryProfile, analyze_query
from rag.retrieval import RetrievedArticle

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}")


def _query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(query):
        token = token.strip()
        if len(token) >= 2 and token not in tokens:
            tokens.append(token)
    return tokens


def _search_text(item: RetrievedArticle) -> str:
    article = item.article
    return " ".join([
        article.law_name,
        article.annotation,
        article.article_text[:300],
    ])


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_equity_query(query_profile: QueryProfile) -> bool:
    return query_profile.topic == "股权善意取得"


def _is_bona_fide_query(query_profile: QueryProfile) -> bool:
    return query_profile.concept == "善意取得"


def _is_base_rule(query_profile: QueryProfile, item: RetrievedArticle) -> bool:
    text = _search_text(item)
    law = item.article.law_name
    if _is_bona_fide_query(query_profile):
        if law == "中华人民共和国民法典" and _has_any(text, ("善意取得", "善意")):
            return True
        if "物权编的解释" in law and _has_any(text, ("善意取得", "善意受让", "时间点")):
            return True
    return False


def _is_topic_specific(query_profile: QueryProfile, item: RetrievedArticle) -> bool:
    text = _search_text(item)
    law = item.article.law_name
    if _is_equity_query(query_profile):
        if item.article.article_num in ("第二十五条", "第二十七条", "第七条"):
            return True
        if any(term in text for term in query_profile.scenario_terms):
            return True
        if _has_any(text, ("股权", "股东", "名义股东", "公司")):
            return True
        if "公司法" in law:
            return True
    return False


def _is_weak_related(query_profile: QueryProfile, item: RetrievedArticle) -> bool:
    if _is_base_rule(query_profile, item) or _is_topic_specific(query_profile, item):
        return False
    return True


def group_topic_candidates(query: str, candidates: list[RetrievedArticle], limit_per_group: int = 8) -> dict[str, list[dict[str, Any]]]:
    query_profile = analyze_query(query)
    groups = {
        "base_rules": [],
        "topic_specific_rules": [],
        "weak_related_rules": [],
    }
    seen: set[tuple[str, str]] = set()

    def append(group_name: str, item: RetrievedArticle, tag: str) -> None:
        key = (item.article.law_id, item.article.article_num)
        if key in seen:
            return
        if len(groups[group_name]) >= limit_per_group:
            return
        groups[group_name].append(
            {
                "law_id": item.article.law_id,
                "law_name": item.article.law_name,
                "article_num": item.article.article_num,
                "annotation": item.article.annotation,
                "group_reason": tag,
                "reasons": sorted(item.reasons),
                "score": item.rerank_score,
            }
        )
        seen.add(key)

    for item in candidates:
        if _is_base_rule(query_profile, item):
            append("base_rules", item, "基础规则")
    for item in candidates:
        if _is_topic_specific(query_profile, item):
            append("topic_specific_rules", item, "专题专门规则")
    for item in candidates:
        if _is_weak_related(query_profile, item):
            append("weak_related_rules", item, "外围弱相关")

    return groups
