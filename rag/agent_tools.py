from __future__ import annotations

"""
Agent tool implementations.

Each function in this module is one "tool" available to the DeepSeek Agent.
They are plain synchronous Python functions that return a formatted string —
easy to test independently and easy to wrap as Function-Calling tool calls.

Tools
-----
hybrid_search          – BM25 + vector + graph expansion (existing pipeline)
lightrag_query         – LightRAG local / mix / global retrieval
get_article            – exact full-text lookup by law name keyword + article num
get_citing_articles    – which articles cite a given article (incoming)
get_cited_articles     – which articles a given article cites (outgoing)
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool 1 — hybrid_search
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    top_k: int = 10,
    law_name_filter: str | None = None,
) -> str:
    """
    Search the law corpus using the full hybrid pipeline:
    BM25 keyword search + dense vector search + citation-graph expansion +
    DeepSeek query analysis + semantic reranking.

    Parameters
    ----------
    query:
        Natural-language legal question.
    top_k:
        Number of results to return (1-10).
    law_name_filter:
        Optional partial law name (e.g. "民法典", "公司法") to restrict
        results to a single statute.

    Returns
    -------
    Formatted string listing the top articles with annotations and scores.
    """
    try:
        import json as _json

        from rag.config import settings
        from rag.service import get_bm25_index, get_index, get_reranker, run_rag_query

        # Resolve law_ids filter
        law_ids: set[str] | None = None
        if law_name_filter:
            law_map: dict[str, str] = _json.loads(
                (settings.law_map_path).read_text(encoding="utf-8")
            )
            matched = {
                lid for lid, lname in law_map.items() if law_name_filter in lname
            }
            if matched:
                law_ids = matched
            else:
                logger.warning(
                    "hybrid_search: no law matched filter %r", law_name_filter
                )

        result = run_rag_query(
            query,
            top_k=min(max(1, top_k), 10),
            graph_expand_k=3,
            law_ids=law_ids,
            compress=False,  # agent will synthesise itself
        )

        lines: list[str] = [f"【hybrid_search 结果】查询：{query}"]
        if law_name_filter:
            lines.append(f"限定法律：{law_name_filter}")
        lines.append("")

        for i, r in enumerate(result["results"], 1):
            reasons = ", ".join(sorted(r["reasons"]))
            # Expose law_id explicitly so the Agent can build [[law_id|...]] citation refs
            lines.append(
                f"[{i}] law_id={r['law_id']}  {r['law_name']} {r['article_num']}"
            )
            if r.get("annotation"):
                lines.append(f"    考点：{r['annotation']}")
            lines.append(f"    检索路径：{reasons}")
            text_snippet = (r.get("article_text") or "")[:300].replace("\n", " ")
            if text_snippet:
                lines.append(f"    正文节选：{text_snippet}…")
            lines.append("")

        qa = result.get("query_analysis", {})
        if qa.get("concept"):
            lines.append(
                f"查询分析 → 概念：{qa['concept']}  主题：{qa.get('topic', '')}"
            )
        if qa.get("expanded_terms"):
            lines.append(f"扩展词：{', '.join(qa['expanded_terms'][:6])}")

        return "\n".join(lines)

    except Exception as exc:
        logger.exception("hybrid_search failed")
        return f"【hybrid_search 错误】{exc}"


# ---------------------------------------------------------------------------
# Tool 2 — lightrag_query
# ---------------------------------------------------------------------------


def lightrag_query(
    question: str,
    mode: str = "mix",
) -> str:
    """
    Query the LightRAG knowledge graph for graph-aware, community-level answers.

    Use this tool when you need:
    - A comprehensive overview of a legal doctrine across multiple statutes
      (mode="global" or mode="hybrid")
    - Precise entity + neighbour context for a known legal concept
      (mode="local")
    - The best general-purpose answer combining graph + vectors
      (mode="mix", the default)

    Parameters
    ----------
    question:
        Natural-language legal question.
    mode:
        One of "local", "global", "hybrid", "mix", "naive".
        Default "mix" is recommended for most queries.

    Returns
    -------
    LightRAG-generated answer string.
    """
    valid_modes = {"local", "global", "hybrid", "mix", "naive"}
    if mode not in valid_modes:
        mode = "mix"

    try:
        from rag.lightrag_bridge import query as lg_query

        answer = lg_query(question, mode=mode)
        return f"【LightRAG ({mode}) 结果】\n{answer}"

    except Exception as exc:
        logger.exception("lightrag_query failed")
        return f"【lightrag_query 错误】{exc}"


# ---------------------------------------------------------------------------
# Tool 3 — get_article
# ---------------------------------------------------------------------------


def get_article(
    law_name_keyword: str,
    article_num: str,
) -> str:
    """
    Retrieve the full text, annotation, chapter, incoming citations, and
    outgoing citations for a specific legal article.

    Use this tool when you already know (or suspect) the exact article you
    need and want to read its complete content.

    Parameters
    ----------
    law_name_keyword:
        Partial or full law name, e.g. "民法典", "公司法若干问题的规定（三）".
        If multiple laws match, the best match is used.
    article_num:
        Article number in Chinese, e.g. "第三百一十一条", "第二十五条".

    Returns
    -------
    Formatted string with the full article content and citation information.
    """
    try:
        import json as _json

        from rag.config import settings
        from rag.loader import build_article_corpus

        law_map: dict[str, str] = _json.loads(
            settings.law_map_path.read_text(encoding="utf-8")
        )

        # Find best-matching law IDs
        matched_ids = {
            lid for lid, lname in law_map.items() if law_name_keyword in lname
        }
        if not matched_ids:
            return f"【get_article 错误】未找到包含 {law_name_keyword!r} 的法律"

        articles = build_article_corpus(law_ids=matched_ids)
        # Normalise article_num for comparison
        target = article_num.strip()
        hits = [a for a in articles if a.article_num == target]

        if not hits:
            # Try fuzzy: check if target is a substring
            hits = [a for a in articles if target in a.article_num]

        if not hits:
            return (
                f"【get_article 错误】在 {law_name_keyword!r} 中未找到条文 {article_num!r}"
                f"可用条文示例：{', '.join(a.article_num for a in articles[:5])}"
            )

        article = hits[0]
        lines: list[str] = [
            f"【{article.law_name}】{article.article_num}",
            f"law_id={article.law_id}",
        ]
        if article.chapter:
            lines.append(f"章节：{article.chapter}")
        if article.annotation:
            lines.append(f"考点标注：{article.annotation}")
        lines.append("")
        lines.append("【全文】")
        lines.append(article.article_text)

        if article.outgoing_citations:
            lines.append("")
            lines.append(
                f"【本条引用的其他法条（{len(article.outgoing_citations)}条）】"
            )
            for edge in article.outgoing_citations[:8]:
                lines.append(
                    f"  → law_id={edge.target_law_id or '?'}  {edge.target_law_name} {edge.target_article}"
                )

        if article.incoming_citations:
            lines.append("")
            lines.append(
                f"【引用本条的其他法条（{len(article.incoming_citations)}条）】"
            )
            for edge in article.incoming_citations[:8]:
                lines.append(
                    f"  ← law_id={edge.source_law_id or '?'}  {edge.source_law_name} {edge.source_article}"
                )

        return "\n".join(lines)

    except Exception as exc:
        logger.exception("get_article failed")
        return f"【get_article 错误】{exc}"


# ---------------------------------------------------------------------------
# Tool 4 — get_citing_articles
# ---------------------------------------------------------------------------


def get_citing_articles(
    law_name_keyword: str,
    article_num: str,
    limit: int = 12,
) -> str:
    """
    Find all articles that cite a given article (incoming citations).

    This is the "who cites this article?" query.  It is particularly useful
    for finding judicial interpretations and implementing regulations that
    apply or refine a core statutory provision.

    Parameters
    ----------
    law_name_keyword:
        Partial law name, e.g. "民法典".
    article_num:
        Article number, e.g. "第三百一十一条".
    limit:
        Maximum number of citing articles to return.

    Returns
    -------
    Formatted list of articles that cite the target article.
    """
    try:
        import json as _json

        from rag.config import settings
        from rag.loader import build_article_corpus

        law_map: dict[str, str] = _json.loads(
            settings.law_map_path.read_text(encoding="utf-8")
        )
        matched_ids = {
            lid for lid, lname in law_map.items() if law_name_keyword in lname
        }
        if not matched_ids:
            return f"【get_citing_articles 错误】未找到法律 {law_name_keyword!r}"

        articles = build_article_corpus(law_ids=matched_ids)
        target_num = article_num.strip()
        hits = [a for a in articles if a.article_num == target_num]
        if not hits:
            return (
                f"【get_citing_articles 错误】在 {law_name_keyword!r} 中"
                f"未找到条文 {article_num!r}"
            )

        article = hits[0]
        incoming = article.incoming_citations[:limit]

        if not incoming:
            return (
                f"【get_citing_articles】{article.law_name}{article.article_num}"
                f" 暂无已知的反向引用"
            )

        lines = [
            f"【引用 {article.law_name}{article.article_num} 的法条"
            f"（共 {len(article.incoming_citations)} 条，显示前 {len(incoming)} 条）】",
            "",
        ]
        for i, edge in enumerate(incoming, 1):
            lines.append(
                f"[{i}] law_id={edge.source_law_id or '?'}  {edge.source_law_name} {edge.source_article}"
            )
            if edge.context:
                snippet = edge.context[:120].replace("\n", " ")
                lines.append(f"     引用上下文：{snippet}…")
            lines.append("")

        return "\n".join(lines)

    except Exception as exc:
        logger.exception("get_citing_articles failed")
        return f"【get_citing_articles 错误】{exc}"


# ---------------------------------------------------------------------------
# Tool 5 — get_cited_articles
# ---------------------------------------------------------------------------


def get_cited_articles(
    law_name_keyword: str,
    article_num: str,
    limit: int = 12,
) -> str:
    """
    Find all articles that a given article cites (outgoing citations).

    This is the "what does this article reference?" query.  Useful for
    tracing the legal basis chain: e.g. which foundational provisions does
    this implementing regulation point back to?

    Parameters
    ----------
    law_name_keyword:
        Partial law name, e.g. "公司法若干问题的规定（三）".
    article_num:
        Article number, e.g. "第二十五条".
    limit:
        Maximum number of cited articles to return.

    Returns
    -------
    Formatted list of articles cited by the target article.
    """
    try:
        import json as _json

        from rag.config import settings
        from rag.loader import build_article_corpus

        law_map: dict[str, str] = _json.loads(
            settings.law_map_path.read_text(encoding="utf-8")
        )
        matched_ids = {
            lid for lid, lname in law_map.items() if law_name_keyword in lname
        }
        if not matched_ids:
            return f"【get_cited_articles 错误】未找到法律 {law_name_keyword!r}"

        articles = build_article_corpus(law_ids=matched_ids)
        target_num = article_num.strip()
        hits = [a for a in articles if a.article_num == target_num]
        if not hits:
            return (
                f"【get_cited_articles 错误】在 {law_name_keyword!r} 中"
                f"未找到条文 {article_num!r}"
            )

        article = hits[0]
        outgoing = article.outgoing_citations[:limit]

        if not outgoing:
            return (
                f"【get_cited_articles】{article.law_name}{article.article_num}"
                f" 正文中未检测到对其他法条的引用"
            )

        lines = [
            f"【{article.law_name}{article.article_num} 引用的法条"
            f"（共 {len(article.outgoing_citations)} 条，显示前 {len(outgoing)} 条）】",
            "",
        ]
        for i, edge in enumerate(outgoing, 1):
            tgt = f"{edge.target_law_name} {edge.target_article}".strip()
            lines.append(f"[{i}] law_id={edge.target_law_id or '?'}  {tgt}")
            if edge.context:
                snippet = edge.context[:120].replace("\n", " ")
                lines.append(f"     引用上下文：{snippet}…")
            lines.append("")

        return "\n".join(lines)

    except Exception as exc:
        logger.exception("get_cited_articles failed")
        return f"【get_cited_articles 错误】{exc}"


# ---------------------------------------------------------------------------
# Tool registry (used by agent.py to build the Function-Calling schema)
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Any] = {
    "hybrid_search": hybrid_search,
    "lightrag_query": lightrag_query,
    "get_article": get_article,
    "get_citing_articles": get_citing_articles,
    "get_cited_articles": get_cited_articles,
}

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": (
                "使用混合检索（BM25关键词+向量语义+引用图扩展）在法条库中搜索相关法律条文。"
                "适合大多数法律问题的首次检索。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自然语言法律问题或检索关键词",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，1-12，默认10",
                        "default": 10,
                    },
                    "law_name_filter": {
                        "type": "string",
                        "description": (
                            "可选。限定在某部法律内检索，如民法典、公司法。"
                            "不确定时留空。"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lightrag_query",
            "description": (
                "使用LightRAG知识图谱进行图感知检索，擅长回答体系性、宏观性法律问题，"
                "如: 某制度在整个法律体系中的地位 / 某概念涉及哪些法律规定。"
                "mode='mix'适合大多数情况；mode='global'适合全局综述性问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "法律问题",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["local", "global", "hybrid", "mix", "naive"],
                        "description": (
                            "检索模式：mix（默认，知识图谱+向量）、"
                            "local（实体邻居）、global（社区摘要）、"
                            "hybrid（local+global）、naive（纯向量）"
                        ),
                        "default": "mix",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_article",
            "description": (
                "精确获取某条法律条文的完整内容，包括全文、考点标注、"
                "章节归属和引用关系。"
                "当你已经知道需要查看哪部法律的哪一条时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "law_name_keyword": {
                        "type": "string",
                        "description": "法律名称关键词，如'民法典'、'公司法若干问题的规定三'",
                    },
                    "article_num": {
                        "type": "string",
                        "description": "条号，如第三百一十一条、第二十五条",
                    },
                },
                "required": ["law_name_keyword", "article_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_citing_articles",
            "description": (
                "查找所有引用了指定法条的其他法条（反向引用/incoming citations）。"
                "常用于从基础条文出发，找到所有参照适用该条的司法解释和实施规定。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "law_name_keyword": {
                        "type": "string",
                        "description": "被引用法律的名称关键词",
                    },
                    "article_num": {
                        "type": "string",
                        "description": "被引用条文的条号",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回数量上限，默认12",
                        "default": 12,
                    },
                },
                "required": ["law_name_keyword", "article_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cited_articles",
            "description": (
                "查找指定法条引用的所有其他法条（正向引用/outgoing citations）。"
                "常用于追溯某条实施规定指向的上位法依据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "law_name_keyword": {
                        "type": "string",
                        "description": "发出引用的法律名称关键词，如'公司法若干问题的规定三'",
                    },
                    "article_num": {
                        "type": "string",
                        "description": "发出引用的条文条号",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回数量上限，默认12",
                        "default": 12,
                    },
                },
                "required": ["law_name_keyword", "article_num"],
            },
        },
    },
]
