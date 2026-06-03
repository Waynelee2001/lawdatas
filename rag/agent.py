from __future__ import annotations

"""
DeepSeek Function-Calling Agent for legal RAG.

Architecture
------------
The agent runs a tight loop (≤ MAX_ROUNDS iterations):

  1. Send the user question + conversation history to DeepSeek with the
     full tool schema attached.
  2. If DeepSeek returns tool_calls  → execute each tool, append results
     to the message history, go to step 1.
  3. If DeepSeek returns plain text  → this is the final answer; return it.
  4. If MAX_ROUNDS is reached without a plain-text response → force a final
     synthesis call with no tools attached.

Tools available (defined in agent_tools.py)
-------------------------------------------
  knowledge_graph_search  staged keyword/vector/BM25 + citation graph retrieval
  hybrid_search         BM25 + vector + graph expansion (fast, precise)
  lightrag_query        LightRAG graph-aware retrieval (comprehensive)
  get_article           Exact full-text lookup by law name + article number
  get_citing_articles   Incoming citation traversal (who cites X?)
  get_cited_articles    Outgoing citation traversal (what does X cite?)
"""

import json
import logging
import os
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

MAX_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "12"))
MAX_TOOL_CALLS_PER_ROUND = int(os.getenv("AGENT_MAX_TOOL_CALLS_PER_ROUND", "0"))
MAX_TOOL_CALLS_TOTAL = int(os.getenv("AGENT_MAX_TOOL_CALLS_TOTAL", "24"))
# <= 0 means keep full tool outputs.  A positive value truncates unusually
# large tool returns to protect slow/free deployments.
TOOL_HISTORY_MAX_CHARS = int(os.getenv("AGENT_TOOL_HISTORY_MAX_CHARS", "0"))
CHAT_MAX_TOKENS = int(os.getenv("AGENT_CHAT_MAX_TOKENS", "3072"))
REQUEST_TIMEOUT = 90.0  # seconds per LLM call

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是专业的法考（中国法律职业资格考试）辅导助手，拥有完整的中国法律法条库检索能力。

## 检索策略（Agent 保持深度，但必须围绕问题核心）

**第零步：先在心里分析问题核心（必做，不要输出过程）**
识别用户真正问的是：核心制度/构成要件/程序阶段/例外规则/司法解释细化/对比辨析中的哪一种。
后续每次工具调用都必须能回答一个明确缺口；如果只是“可能相关”，不要继续查。

**第一步：知识图谱优先检索（必做，默认首选）**
用 knowledge_graph_search 做首轮检索：
- seed_top_k=15
- graph_depth=2
- graph_expand_k=4

这个工具会先用关键词、BM25、向量定位核心条文，再沿引用图谱查：
核心条文 → 关联司法解释/配套规定 → 司法解释再关联的其他法条。
一般问题首轮结果足够时，直接综合作答。

**第二步：覆盖检查（每次补查前必做）**
检查首轮图谱是否已经覆盖：
1. 基本法律的核心条文
2. 主要司法解释或配套规范
3. 重要例外/排除/程序衔接
4. 与用户问题直接相关的场景关键词

若以上核心覆盖已经够，不要继续调用工具，直接作答。

**第三步：定向深挖（按缺口使用，不是漫游）**
- 缺少某条完整规范含义：get_article；通常只读最关键的 2-5 条，不要机械读取首轮所有候选
- 缺少“谁细化/参照适用此条”：get_citing_articles
- 缺少“该解释指向哪些上位依据”：get_cited_articles
- 首轮核心命中明显偏离：再用 hybrid_search 换关键词补检索
- 体系性、宏观概念仍不清楚：才用 lightrag_query

复杂体系性问题可以继续深挖，但每一轮必须围绕用户问题核心收敛；
一旦发现后续材料与问题核心相关性下降，应停止检索并综合作答。

## 工具选用场景
- knowledge_graph_search：默认首轮工具，关键词+向量+BM25+引用知识图谱扩展
- hybrid_search：补充检索，top_k 通常设为 10-15
- get_citing_articles：核心法条 → 司法解释和配套规定
- get_article：读完整条文，获取引用关系
- get_cited_articles：追溯某条引用的上位法依据
- lightrag_query：体系性综述，mode="mix" 或 "global"

## 法条引用格式（严格遵守，无一例外）

在最终答案中，作为正式依据展开的法律条文引用，必须使用以下格式：
  [[law_id|法律名称|条号|标注]]

示例：
  [[3346|中华人民共和国民法典|第三百一十一条|善意取得]]
  [[250|最高人民法院关于适用《中华人民共和国公司法》若干问题的规定（三）|第二十五条|名义股东处分股权的效力]]

规则：
1. law_id 必须从工具返回结果的 law_id=XXXX 字段直接提取，不得猜测或编造
2. 工具结果中每条法条都显示 law_id=数字，直接复制该数字
3. 标注字段可留空；留空时格式必须写成 [[law_id|法律名称|条号|]]，不要写成两个连续竖线
4. 核心依据和直接用于推理的条文必须用 [[...]] 格式
5. “条号”字段必须是具体条文，例如“第五十六条”“第三百一十一条”；章节、编、章、节、款、项、司法解释标题、制度标题不得写入 [[...]] 占位符
6. 不要为了给外围、顺带、弱相关条文补链接而继续检索；如果没有 law_id 或具体条号，就不要把该内容作为可点击正式依据展开
7. 禁止使用"《民法典》第X条"这种裸文本引用方式作为核心依据

## 回答要求
- 先给出明确结论，再展开要件/情形分析
- 法条引用嵌入分析句子内部（不要单列法条清单）
- 覆盖主要规定、重要司法解释细化、例外情形、程序衔接
- 内容完整，优先列核心规则和必要配套规定；不要堆砌外围弱相关条文
- 可以使用 Markdown 标题、加粗、列表、引用和必要的短表格
- 禁止普通代码块和 ASCII/字符画图示，答案中不得出现“┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ ▼ ▲”等方框、横线或箭头拼出来的图；如确需体系图或流程图，只能输出以 ```mermaid 开头的 Mermaid 图，前端会渲染成图
- 不要输出“信息已经足够”“我来回答”等检索过程性句子，直接给答案
- 适合法考备考场景，条理清晰
- 所有条号必须来自工具检索结果，不得凭记忆填写
"""

_FORCE_FINAL_PROMPT = """\
请根据以上所有工具检索结果，给出完整、准确的最终答案。

严格要求：
1. 所有法条引用必须用 [[law_id|法律名称|条号|标注]] 格式，law_id 从工具结果的 law_id= 字段直接提取
2. 核心依据和直接用于推理的条文必须用 [[...]] 格式，禁止裸文本引用
3. 只有具体“第X条”才能写成 [[...]]；“第X章”“第X节”“第四章第七节”这类章节层级只能作为普通文字说明，不能做链接
4. 标注字段留空时只能写成 [[law_id|法律名称|条号|]]，不得写成 [[law_id|法律名称|条号||]]
5. 区分：主要依据（核心法条）/ 司法解释细化 / 例外情形 / 程序衔接
6. 先给结论，后展开分析，覆盖所有重要法条和司法解释
7. 内容完整，优先呈现法考复习最需要的核心体系
8. 可以使用 Markdown 标题、加粗、列表、引用和必要的短表格
9. 禁止普通代码块和 ASCII/字符画图示，答案中不得出现“┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ ▼ ▲”等方框、横线或箭头拼出来的图；如确需图示，只能输出以 ```mermaid 开头的 Mermaid 图
10. 不要输出“信息已经足够”“我来回答”等检索过程性句子，直接给答案
11. 如果已经覆盖核心基本法条、主要司法解释/配套规范和程序衔接，应立即综合作答，不要继续穷尽外围条款
12. 围绕用户问题核心取舍材料；图谱中弱相关、远相关节点不得展开
"""

_BOUNDED_FINAL_PROMPT = """\
你是专业的法考（中国法律职业资格考试）辅导助手。请只根据用户问题和给出的检索结果作答。

法条引用格式：
- 正式依据必须写成 [[law_id|法律名称|条号|标注]]
- 第一个字段只能是数字，例如 [[13|中华人民共和国刑事诉讼法|第五十六条|非法证据排除]]；禁止写成 law_id=13
- 标注字段留空时写成 [[13|中华人民共和国刑事诉讼法|第五十六条|]]，不要写成两个连续竖线
- law_id、法律名称、条号必须从检索结果中复制，不得猜测
- 只有具体“第X条”才能写成 [[...]]；章节、编、章、节、制度标题只能作为普通文字说明
- 如果只是概括说明但没有 law_id，不要写成裸条文依据

回答要求：
- 先给明确结论，再分点说明
- 检索结果已经按优先级排列；[1] 通常是首要主依据，除非明显无关，必须首先使用 [1]
- 若检索结果同时包含基本法律、司法解释和规范性文件，优先以基本法律条文作为主依据，再用司法解释/规范性文件细化
- 优先覆盖核心规则、程序规则、重要例外和司法解释细化
- 控制在 800-1200 个中文字符
- 可以使用 Markdown 标题、加粗、列表、引用和必要的短表格
- 禁止普通代码块和 ASCII/字符画图示，答案中不得出现“┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ ▼ ▲”等方框、横线或箭头拼出来的图；如确需图示，只能输出以 ```mermaid 开头的 Mermaid 图
- 不要输出检索过程性句子，直接给答案
"""


# ---------------------------------------------------------------------------
# Core agent class
# ---------------------------------------------------------------------------


class LegalAgent:
    """
    DeepSeek Function-Calling agent with access to the law corpus tools.

    Usage
    -----
        agent = LegalAgent()
        result = agent.run("股权善意取得的构成要件是什么？")
        print(result["answer"])
    """

    def __init__(self) -> None:
        from rag.agent_tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
        from rag.config import settings

        self._api_key = settings.chat_api_key
        self._base_url = settings.chat_base_url.rstrip("/")
        self._model = settings.chat_model
        self._tools = TOOL_SCHEMAS
        self._tool_fns = TOOL_FUNCTIONS
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> dict:
        """Send a chat completion request; return the full response dict."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens or CHAT_MAX_TOKENS,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = self._session.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _execute_tool(self, name: str, arguments_json: str) -> str:
        """Parse arguments and call the corresponding tool function."""
        fn = self._tool_fns.get(name)
        if fn is None:
            return f"【错误】未知工具：{name}"
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as exc:
            return f"【错误】工具参数解析失败（{name}）：{exc}"
        try:
            logger.info("Calling tool %r with args %s", name, args)
            t0 = time.perf_counter()
            result = fn(**args)
            elapsed = time.perf_counter() - t0
            logger.info("Tool %r returned in %.2fs", name, elapsed)
            return str(result)
        except Exception as exc:
            logger.exception("Tool %r raised an exception", name)
            return f"【工具执行错误 ({name})】{exc}"

    def _compact_tool_output(self, output: str) -> str:
        """Keep tool context small enough for Render Free and fast LLM turns."""
        text = str(output or "")
        if TOOL_HISTORY_MAX_CHARS <= 0:
            return text
        if len(text) <= TOOL_HISTORY_MAX_CHARS:
            return text
        return (
            text[:TOOL_HISTORY_MAX_CHARS]
            + "\n\n【提示】以上工具结果已截断；请基于已给出的 law_id、法名、条号、考点和正文节选作答。"
        )

    def _normalize_answer_refs(self, answer: str) -> str:
        """Repair common citation bracket slips so the frontend can hyperlink."""
        text = str(answer or "")
        text = re.sub(
            r"^\s*(好的[，,。]?\s*)?(?:现在|下面|接下来)?\s*"
            r"(?:信息|材料|检索结果|证据)(?:已经|已)?(?:非常)?(?:充分|足够|完整)"
            r"[，,。]?\s*(?:可以|能够)?(?:给出|作出)?(?:完整|准确|最终)?(?:准确)?"
            r"(?:的)?(?:回答|答案|结论)了?[。:：]?\s*",
            "",
            text,
            flags=re.S,
        )
        text = re.sub(
            r"^\s*(好的[，,。]?\s*)?(?:现在)?\s*"
            r"(?:信息|材料|检索结果|证据)(?:已经|已)?(?:非常)?"
            r"(?:充分|足够|完整|全面)了?[。:：]?\s*"
            r"(?:让我|下面|接下来)?(?:整理|给出|作出).*?(?:回答|答案|结论)[。:：]?\s*",
            "",
            text,
            flags=re.S,
        )
        text = re.sub(
            r"^\s*(好的[，,。]?\s*)?(?:信息|材料|检索结果|证据)(?:已经|已)?(?:非常)?"
            r"(?:充分|足够|完整).*?(?:现在|下面|接下来)?(?:来)?回答[。:：]?\s*",
            "",
            text,
            flags=re.S,
        )
        text = re.sub(r"^\s*(好的[，,。]?\s*)?(?:现在|下面|接下来)(?:来)?回答[。:：]?\s*", "", text)
        text = re.sub(r"^\s*---+\s*", "", text)
        text = re.sub(r"\[\[([^\]\n]{1,260}?)[】］](?!\])", r"[[\1]]", text)
        text = re.sub(r"\[\[\s*law_id\s*=\s*(\d+)\s*\|", r"[[\1|", text)
        text = re.sub(
            r"\[\[([^|\]\n]+)\|([^|\]\n]+)\|([^|\]\n]+)\]\]",
            r"[[\1|\2|\3|]]",
            text,
        )
        text = re.sub(
            r"\[\[(\d+)\|([^|\]\n]+)\|([^|\]\n]+)\|([^\]\n]*)\]\]",
            self._normalize_ref_placeholder,
            text,
        )
        return self._strip_ascii_diagrams(text)

    def _strip_ascii_diagrams(self, answer: str) -> str:
        """Remove box-drawing diagrams that break the narrow AI sidebar."""
        diagram_re = re.compile(r"[┌┐└┘├┤┬┴┼─━│┃▼▲]")
        lines = str(answer or "").splitlines()
        cleaned: list[str] = []
        in_diagram = False
        replacement = (
            "体系关系：核心规则、出资义务和人格否认例外请以上文分点说明为准。"
        )

        for line in lines:
            if diagram_re.search(line):
                if not in_diagram:
                    cleaned.append(replacement)
                in_diagram = True
                continue
            if in_diagram and not line.strip():
                continue
            in_diagram = False
            cleaned.append(line)

        return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()

    def _normalize_ref_placeholder(self, match: re.Match[str]) -> str:
        law_id = match.group(1).strip()
        law_name = match.group(2).strip()
        article_num = match.group(3).strip()
        annotation = match.group(4).strip().strip("|")
        if self._is_linkable_article_num(article_num):
            return f"[[{law_id}|{law_name}|{article_num}|{annotation}]]"
        label = f"《{law_name}》{article_num}"
        if annotation:
            label += f"【{annotation}】"
        return label

    def _auto_link_known_refs(self, answer: str, messages: list[dict]) -> str:
        """
        Convert bare full-name law references into frontend link placeholders.

        The model sometimes follows the legal wording but forgets the [[...]]
        wrapper, e.g. 《中华人民共和国刑事诉讼法》第五十六条.  Tool outputs
        already contain the trusted law_id/law_name/article_num triples, so we
        can safely link exact full-name references without guessing ids.
        """
        text = str(answer or "")
        refs = self._extract_known_refs(messages)
        if not refs:
            return text

        for ref in sorted(
            refs,
            key=lambda item: len(item["law_name"]) + len(item["article_num"]),
            reverse=True,
        ):
            law_id = ref["law_id"]
            law_name = ref["law_name"]
            article_num = ref["article_num"]
            annotation = ref.get("annotation", "")
            placeholder = f"[[{law_id}|{law_name}|{article_num}|{annotation}]]"
            variants = [
                f"《{law_name}》{article_num}",
                f"《{law_name}》 {article_num}",
                f"{law_name}{article_num}",
                f"{law_name} {article_num}",
            ]
            for variant in variants:
                if variant and variant in text and placeholder not in text:
                    text = text.replace(variant, placeholder)
        return self._normalize_answer_refs(text)

    def _extract_known_refs(self, messages: list[dict]) -> list[dict[str, str]]:
        article_re = (
            r"第[一二三四五六七八九十百千万亿零〇○0-9]+条"
            r"(?:之[一二三四五六七八九十百千万亿零〇○0-9]+)?"
        )
        refs: dict[tuple[str, str, str], dict[str, str]] = {}
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = str(message.get("content") or "")
            for match in re.finditer(
                rf"law_id=(\d+)\s+(.+?)\s+({article_re})(?=\s|$|【)",
                content,
            ):
                law_id, law_name, article_num = (
                    match.group(1).strip(),
                    match.group(2).strip(),
                    match.group(3).strip(),
                )
                refs.setdefault(
                    (law_id, law_name, article_num),
                    {
                        "law_id": law_id,
                        "law_name": law_name,
                        "article_num": article_num,
                        "annotation": "",
                    },
                )
            for match in re.finditer(
                rf"【([^】\n]+)】({article_re})\s*\nlaw_id=(\d+)",
                content,
            ):
                law_name, article_num, law_id = (
                    match.group(1).strip(),
                    match.group(2).strip(),
                    match.group(3).strip(),
                )
                refs.setdefault(
                    (law_id, law_name, article_num),
                    {
                        "law_id": law_id,
                        "law_name": law_name,
                        "article_num": article_num,
                        "annotation": "",
                    },
                )
        return list(refs.values())

    @staticmethod
    def _is_linkable_article_num(article_num: str) -> bool:
        return bool(
            re.fullmatch(
                r"第[一二三四五六七八九十百千万亿零〇○0-9]+条(?:之[一二三四五六七八九十百千万亿零〇○0-9]+)?",
                str(article_num or "").strip(),
            )
        )

    def _prioritize_bounded_results(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        law_hints = [
            (("刑事诉讼法", "刑诉法"), "中华人民共和国刑事诉讼法"),
            (("民事诉讼法", "民诉法"), "中华人民共和国民事诉讼法"),
            (("民法典",), "中华人民共和国民法典"),
            (("公司法",), "中华人民共和国公司法"),
            (("行政诉讼法",), "中华人民共和国行政诉讼法"),
            (("刑法",), "中华人民共和国刑法"),
        ]
        explicit_law = ""
        for aliases, law_name in law_hints:
            if any(alias in query for alias in aliases):
                explicit_law = law_name
                break

        def norm_name(item: dict[str, Any]) -> str:
            return str(item.get("law_name", "") or "").strip("《》")

        def rank(item: dict[str, Any]) -> tuple[float, float]:
            name = norm_name(item)
            annotation = str(item.get("annotation", "") or "")
            score = float(item.get("rerank_score") or item.get("score") or 0)
            boost = 0.0
            if explicit_law and name == explicit_law:
                boost += 100.0
            elif explicit_law and explicit_law in name:
                boost += 40.0
            elif name.startswith("中华人民共和国"):
                boost += 12.0
            if annotation and annotation in query:
                boost += 15.0
            if "解释" in name:
                boost += 5.0
            return (boost, score)

        return sorted(results, key=rank, reverse=True)

    def _format_bounded_context(self, rag_data: dict[str, Any]) -> str:
        lines = ["【已检索到的核心法条（按优先级排列，结果1为首要主依据）】"]
        results = self._prioritize_bounded_results(
            str(rag_data.get("query", "") or ""), rag_data.get("results", []) or []
        )
        for i, item in enumerate(results[:15], 1):
            law_id = str(item.get("law_id", "")).strip()
            law_name = str(item.get("law_name", "")).strip()
            article_num = str(item.get("article_num", "")).strip()
            annotation = str(item.get("annotation", "") or "").strip()
            ref = f"[[{law_id}|{law_name}|{article_num}|{annotation}]]"
            reasons = ", ".join(sorted(item.get("reasons", []) or []))
            article_text = str(item.get("article_text", "") or "").replace("\n", " ")
            prefix = "【首要主依据】" if i == 1 else ""
            lines.append(f"结果{i}：{prefix}引用格式 {ref}")
            if annotation:
                lines.append(f"    考点：{annotation}")
            if reasons:
                lines.append(f"    检索路径：{reasons}")
            if article_text:
                lines.append(f"    正文：{article_text[:360]}")
            incoming = item.get("incoming_citations", []) or []
            if incoming:
                snippets = []
                for edge in incoming[:2]:
                    src_id = str(edge.get("source_law_id", "") or "?")
                    src_name = str(edge.get("source_law_name", "") or "")
                    src_article = str(edge.get("source_article", "") or "")
                    if src_id != "?" and src_name and src_article:
                        snippets.append(f"[[{src_id}|{src_name}|{src_article}|]]")
                if snippets:
                    lines.append("    相关引用：" + "；".join(snippets))
            lines.append("")
        return "\n".join(lines).strip()

    def _fallback_bounded_answer(self, rag_data: dict[str, Any]) -> str:
        lines = ["## 结论", "已检索到以下核心依据，可作为本题作答框架：", ""]
        for item in rag_data.get("results", [])[:5]:
            law_id = str(item.get("law_id", "")).strip()
            law_name = str(item.get("law_name", "")).strip()
            article_num = str(item.get("article_num", "")).strip()
            annotation = str(item.get("annotation", "") or "").strip()
            article_text = str(item.get("article_text", "") or "").replace("\n", " ")
            ref = f"[[{law_id}|{law_name}|{article_num}|{annotation}]]"
            summary = article_text[:180]
            lines.append(f"- {ref}：{summary}")
        return self._normalize_answer_refs("\n".join(lines))

    def _ensure_primary_ref(self, answer: str, rag_data: dict[str, Any]) -> str:
        results = self._prioritize_bounded_results(
            str(rag_data.get("query", "") or ""), rag_data.get("results", []) or []
        )
        if not results:
            return answer
        item = results[0]
        law_id = str(item.get("law_id", "") or "").strip()
        law_name = str(item.get("law_name", "") or "").strip()
        article_num = str(item.get("article_num", "") or "").strip()
        annotation = str(item.get("annotation", "") or "").strip()
        if not law_id or not law_name or not article_num:
            return answer
        marker = f"[[{law_id}|{law_name}|{article_num}"
        if marker in answer:
            return answer
        ref = f"[[{law_id}|{law_name}|{article_num}|{annotation}]]"
        return f"核心依据：{ref}。\n\n{answer}"

    def _run_bounded(
        self,
        question: str,
        *,
        verbose: bool = False,
    ) -> dict[str, Any]:
        from rag.service import run_rag_query

        t0 = time.perf_counter()
        rag_data = run_rag_query(
            question,
            top_k=15,
            graph_expand_k=4,
            compress=False,
            use_llm_analysis=False,
        )
        context = self._format_bounded_context(rag_data)
        tool_call_log = [
            {
                "round": 1,
                "tool": "bounded_hybrid_search",
                "args": json.dumps(
                    {"query": question, "top_k": 15, "graph_expand_k": 4},
                    ensure_ascii=False,
                ),
                "output_preview": context[:300],
                "output_chars": len(context),
                "history_chars": len(context),
            }
        ]
        messages = [
            {"role": "system", "content": _BOUNDED_FINAL_PROMPT},
            {
                "role": "user",
                "content": f"用户问题：{question}\n\n{context}\n\n请给出最终答案。",
            },
        ]
        try:
            response = self._chat(messages, tools=None, max_tokens=1200, timeout=45.0)
            answer = response["choices"][0]["message"].get("content") or ""
            answer = self._normalize_answer_refs(answer)
            answer = self._ensure_primary_ref(answer, rag_data)
            messages.append(response["choices"][0]["message"])
        except Exception as exc:
            logger.warning("Bounded agent final synthesis failed (%s); using fallback.", exc)
            answer = self._fallback_bounded_answer(rag_data)
            answer = self._ensure_primary_ref(answer, rag_data)

        if verbose:
            logger.info("Bounded agent finished in %.2fs.", time.perf_counter() - t0)
        return {
            "answer": answer,
            "rounds": 1,
            "tool_calls": tool_call_log,
            "messages": messages,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Run the agent on *question* and return a result dict.

        Result keys
        -----------
        answer      : str   — final answer text
        rounds      : int   — number of tool-call rounds used
        tool_calls  : list  — log of all tool calls and their outputs
        messages    : list  — full message history (for debugging)
        """
        if os.getenv("AGENT_BOUNDED", "0") == "1":
            return self._run_bounded(question, verbose=verbose)

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_call_log: list[dict] = []
        rounds = 0
        force_reason = ""

        while rounds < MAX_ROUNDS:
            rounds += 1
            if verbose:
                logger.info("--- Agent round %d ---", rounds)

            response = self._chat(messages, tools=self._tools)
            choice = response["choices"][0]
            message = choice["message"]

            # Append assistant message to history
            messages.append(message)

            # ---- Check if DeepSeek wants to call tools ----
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                # Plain-text response → final answer
                answer = message.get("content") or ""
                answer = self._normalize_answer_refs(answer)
                answer = self._auto_link_known_refs(answer, messages)
                if verbose:
                    logger.info("Agent finished in %d round(s).", rounds)
                return {
                    "answer": answer,
                    "rounds": rounds,
                    "tool_calls": tool_call_log,
                    "messages": messages,
                }

            # ---- Execute each tool call ----
            for idx, tc in enumerate(tool_calls):
                tool_name = tc["function"]["name"]
                tool_args = tc["function"].get("arguments", "{}")
                tool_call_id = tc["id"]

                if verbose:
                    logger.info("  → %s(%s)", tool_name, tool_args[:120])

                if (
                    MAX_TOOL_CALLS_TOTAL > 0
                    and len(tool_call_log) >= MAX_TOOL_CALLS_TOTAL
                ):
                    output = (
                        "【工具调用已跳过】已达到本次检索的工具调用总量保护线；"
                        "请基于已返回的核心法条和图谱链路作答。"
                    )
                elif MAX_TOOL_CALLS_PER_ROUND > 0 and idx >= MAX_TOOL_CALLS_PER_ROUND:
                    output = (
                        "【工具调用已跳过】本轮已达到工具调用上限；请先基于已返回的核心法条作答，"
                        "只有信息明显不足时再发起下一轮更精确的检索。"
                    )
                else:
                    output = self._execute_tool(tool_name, tool_args)
                compact_output = self._compact_tool_output(output)

                tool_call_log.append(
                    {
                        "round": rounds,
                        "tool": tool_name,
                        "args": tool_args,
                        "output_preview": output[:300],
                        "output_chars": len(output),
                        "history_chars": len(compact_output),
                    }
                )

                # Append tool result to history
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": compact_output,
                    }
                )

            if (
                MAX_TOOL_CALLS_TOTAL > 0
                and len(tool_call_log) >= MAX_TOOL_CALLS_TOTAL
            ):
                force_reason = f"MAX_TOOL_CALLS_TOTAL ({MAX_TOOL_CALLS_TOTAL})"
                break

        # ---- Guardrail reached: force a final synthesis ----
        if not force_reason:
            force_reason = f"MAX_ROUNDS ({MAX_ROUNDS})"
        logger.warning("%s reached; forcing final synthesis.", force_reason)
        messages.append({"role": "user", "content": _FORCE_FINAL_PROMPT})
        response = self._chat(messages, tools=None, max_tokens=3072)  # no tools → must answer
        answer = response["choices"][0]["message"].get("content") or ""
        answer = self._normalize_answer_refs(answer)
        answer = self._auto_link_known_refs(answer, messages)
        messages.append(response["choices"][0]["message"])

        return {
            "answer": answer,
            "rounds": rounds,
            "tool_calls": tool_call_log,
            "messages": messages,
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


_agent_singleton: LegalAgent | None = None


def get_agent() -> LegalAgent:
    """Return a cached LegalAgent instance (one per process)."""
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = LegalAgent()
    return _agent_singleton


def run_agent_query(
    question: str,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Top-level entry point used by auth_server.py.

    Parameters
    ----------
    question:
        The user's legal question.
    verbose:
        If True, emit INFO-level logs for each agent round and tool call.

    Returns
    -------
    dict with keys: answer, rounds, tool_calls, messages
    """
    return get_agent().run(question, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run LegalAgent on a question")
    parser.add_argument("question", help="Legal question to ask")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool calls")
    args = parser.parse_args()

    result = run_agent_query(args.question, verbose=args.verbose)

    print("\n" + "=" * 60)
    print("【最终答案】")
    print("=" * 60)
    print(result["answer"])
    print()
    print(f"使用轮次：{result['rounds']} / {MAX_ROUNDS}")
    if result["tool_calls"]:
        print(f"\n工具调用记录（{len(result['tool_calls'])} 次）：")
        for i, tc in enumerate(result["tool_calls"], 1):
            print(f"  [{i}] Round {tc['round']}  {tc['tool']}")
            try:
                args_dict = json.loads(tc["args"])
                for k, v in args_dict.items():
                    print(f"       {k}: {str(v)[:80]}")
            except Exception:
                print(f"       args: {tc['args'][:80]}")
