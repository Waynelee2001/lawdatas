from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorRef:
    law_name: str
    article_num: str


@dataclass(frozen=True)
class QueryProfile:
    raw_query: str
    query_type: str = "general"
    concept: str = ""
    asset_type: str = ""
    topic: str = ""
    expanded_terms: tuple[str, ...] = field(default_factory=tuple)
    scenario_terms: tuple[str, ...] = field(default_factory=tuple)
    anchor_refs: tuple[AnchorRef, ...] = field(default_factory=tuple)
    preferred_law_keywords: tuple[str, ...] = field(default_factory=tuple)
    suppress_law_keywords: tuple[str, ...] = field(default_factory=tuple)
    llm_analyzed: bool = False

    @property
    def expanded_query(self) -> str:
        tokens: list[str] = [self.raw_query]
        for term in self.expanded_terms:
            if term and term not in self.raw_query and term not in tokens:
                tokens.append(term)
        return " ".join(tokens)


# ---------------------------------------------------------------------------
# Rule-based fallback (original logic, kept as-is for reliability)
# ---------------------------------------------------------------------------


def analyze_query(query: str) -> QueryProfile:
    """
    Fast, synchronous, rule-based query analysis.
    Used as a fallback when LLM analysis is unavailable or fails.
    """
    q = (query or "").strip()
    query_type = (
        "listing"
        if any(
            token in q
            for token in ("哪些", "有哪些", "规定", "法条", "司法解释", "情形")
        )
        else "general"
    )

    if "股权" in q and ("善意取得" in q or ("善意" in q and "取得" in q)):
        return QueryProfile(
            raw_query=q,
            query_type=query_type,
            concept="善意取得",
            asset_type="股权",
            topic="股权善意取得",
            expanded_terms=(
                "民法典第三百一十一条",
                "股权转让",
                "处分股权",
                "名义股东",
                "无处分权财产出资",
                "原股东处分股权",
                "公司法规定（三）",
                "公司法解释（三）",
            ),
            scenario_terms=(
                "股权转让",
                "处分股权",
                "名义股东",
                "原股东处分股权",
                "无处分权财产出资",
                "公司",
                "股东",
                "一股二卖",
            ),
            anchor_refs=(AnchorRef("中华人民共和国民法典", "第三百一十一条"),),
            preferred_law_keywords=("公司法", "民法典", "物权编", "最高人民法院"),
            suppress_law_keywords=("破产法",),
        )

    if "强制性规定" in q and ("无效" in q or "民事法律行为" in q):
        return QueryProfile(
            raw_query=q,
            query_type=query_type,
            concept="违反强制性规定的民事法律行为效力",
            topic="民事法律行为无效",
            expanded_terms=(
                "民法典第一百五十三条",
                "民事法律行为无效",
                "违背公序良俗",
                "合同编通则解释",
                "民间借贷司法解释",
            ),
            scenario_terms=("无效", "强制性规定", "公序良俗", "合同", "民间借贷"),
            anchor_refs=(AnchorRef("中华人民共和国民法典", "第一百五十三条"),),
            preferred_law_keywords=("民法典", "最高人民法院"),
        )

    return QueryProfile(raw_query=q, query_type=query_type)


# ---------------------------------------------------------------------------
# LLM-powered analysis
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
你是法律检索专家助手，专门服务于中国法律职业资格考试（法考）场景。
用户会提出法律相关检索问题，你需要分析问题并提取关键检索信息。

输出必须是合法的 JSON 对象，包含以下字段（全部必填，不明确时填空字符串或空数组）：

{
  "query_type": "listing" 或 "general",
  "concept": "核心法律概念，如'善意取得'、'表见代理'、'缔约过失责任'",
  "topic": "更具体的话题描述，如'股权善意取得'、'无权处分合同效力'",
  "expanded_terms": ["检索扩展词，包含同义词、相关术语、具体条号引用"],
  "scenario_terms": ["典型场景关键词，用于在法条正文中做术语匹配"],
  "preferred_laws": ["优先检索的法律名称关键词，如'民法典'、'公司法'、'最高人民法院'"],
  "suppress_laws": ["与本问题基本无关的法律关键词，检索时降权，如'破产法'、'海商法'"],
  "anchor_refs": [
    {"law_name": "明确提到的法律全称", "article_num": "明确提到的条号，如'第三百一十一条'"}
  ]
}

规则：
- query_type：若问题含"哪些""有哪些""情形""规定""司法解释""法条"等列举性词汇，填 "listing"；否则填 "general"。
- expanded_terms：尽量提供 4~10 个，覆盖同义词、上位概念、下位概念、常见考点表述、相关条号。
- scenario_terms：提供 3~8 个在法条正文或标注里可能出现的关键词。
- anchor_refs：只有用户问题中明确点名某部法律某条时才填写；不要凭空猜测条号。
- suppress_laws：只填与本题明显无关的法律，不要乱填。
- 不要在 JSON 之外输出任何其他内容。
"""

_LLM_USER_TEMPLATE = "请分析以下法律检索问题：\n\n{query}"


def analyze_query_with_llm(
    query: str,
    *,
    api_key: str,
    base_url: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    timeout: float = 15.0,
) -> QueryProfile:
    """
    Call SiliconFlow chat API to extract a rich QueryProfile for any legal query.
    Falls back to rule-based analysis on any error.
    """
    q = (query or "").strip()
    if not q:
        return analyze_query(q)

    if not api_key:
        logger.debug("No API key — skipping LLM query analysis")
        return analyze_query(q)

    try:
        payload = {
            "model": model_name,
            "temperature": 0.1,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": _LLM_USER_TEMPLATE.format(query=q)},
            ],
        }
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return _build_profile_from_llm(q, parsed)

    except Exception as exc:
        logger.warning(
            "LLM query analysis failed (%s), falling back to rule-based.", exc
        )
        return analyze_query(q)


def _build_profile_from_llm(raw_query: str, parsed: dict) -> QueryProfile:
    """Convert LLM JSON output into a QueryProfile, with safe defaults."""

    def _str(key: str) -> str:
        return str(parsed.get(key) or "").strip()

    def _str_list(key: str) -> tuple[str, ...]:
        raw = parsed.get(key) or []
        if not isinstance(raw, list):
            return ()
        return tuple(str(item).strip() for item in raw if str(item).strip())

    # Parse anchor_refs carefully
    raw_anchors = parsed.get("anchor_refs") or []
    anchors: list[AnchorRef] = []
    if isinstance(raw_anchors, list):
        for item in raw_anchors:
            if isinstance(item, dict):
                law_name = str(item.get("law_name") or "").strip()
                article_num = str(item.get("article_num") or "").strip()
                if law_name and article_num:
                    anchors.append(
                        AnchorRef(law_name=law_name, article_num=article_num)
                    )

    query_type = _str("query_type")
    if query_type not in ("listing", "general"):
        # Re-derive from raw query as safety net
        query_type = (
            "listing"
            if any(
                t in raw_query
                for t in ("哪些", "有哪些", "规定", "法条", "司法解释", "情形")
            )
            else "general"
        )

    return QueryProfile(
        raw_query=raw_query,
        query_type=query_type,
        concept=_str("concept"),
        topic=_str("topic"),
        expanded_terms=_str_list("expanded_terms"),
        scenario_terms=_str_list("scenario_terms"),
        anchor_refs=tuple(anchors),
        preferred_law_keywords=_str_list("preferred_laws"),
        suppress_law_keywords=_str_list("suppress_laws"),
        llm_analyzed=True,
    )
