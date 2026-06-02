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
  hybrid_search         BM25 + vector + graph expansion (fast, precise)
  lightrag_query        LightRAG graph-aware retrieval (comprehensive)
  get_article           Exact full-text lookup by law name + article number
  get_citing_articles   Incoming citation traversal (who cites X?)
  get_cited_articles    Outgoing citation traversal (what does X cite?)
"""

import json
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

MAX_ROUNDS = 7  # hard cap on tool-call rounds
REQUEST_TIMEOUT = 90.0  # seconds per DeepSeek call

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是专业的法考（中国法律职业资格考试）辅导助手，拥有完整的中国法律法条库检索能力。

## 检索策略（必须遵守，至少完成前三步）

**第一步：broad search（必做）**
用 hybrid_search 做初步检索，top_k=10，找到核心法条及相关法条。

**第二步：drill down - 反向引用（必做）**
对命中的每一个核心法条，立刻用 get_citing_articles 查反向引用。
这是找司法解释、实施规定的唯一可靠途径。
例如找到民法典第X条 → 必须查"哪些法条引用了它" → 获得所有参照适用的司法解释。

**第三步：read full text（必做）**
对最重要的 2-4 条法条，用 get_article 读完整正文，确认细节和引用关系。

**第四步：补充检索（建议）**
如覆盖不完整，再做一次 hybrid_search 或 lightrag_query（mode="mix"）。

复杂体系性问题应完成 4-6 轮工具调用，不要在信息不足时就草草作答。

## 工具选用场景
- hybrid_search：初步检索、补充检索，top_k 设为 10
- get_citing_articles：核心法条 → 司法解释（每个核心法条都必须查一次！）
- get_article：读完整条文，获取引用关系
- get_cited_articles：追溯某条引用的上位法依据
- lightrag_query：体系性综述，mode="mix" 或 "global"

## 法条引用格式（严格遵守，无一例外）

在最终答案中，所有对法律条文的引用，无论是核心法条还是顺带提及的条文，
都必须使用以下格式：
  [[law_id|法律名称|条号|标注]]

示例：
  [[3346|中华人民共和国民法典|第三百一十一条|善意取得]]
  [[250|最高人民法院关于适用《中华人民共和国公司法》若干问题的规定（三）|第二十五条|名义股东处分股权的效力]]

规则：
1. law_id 必须从工具返回结果的 law_id=XXXX 字段直接提取，不得猜测或编造
2. 工具结果中每条法条都显示 law_id=数字，直接复制该数字
3. 标注字段可留空（写||即可），但 law_id、法律名称、条号三项必须准确
4. 凡是提到法律条文，即使只是一笔带过，也必须用 [[...]] 格式
5. 禁止使用"《民法典》第X条"这种裸文本引用方式

## 回答要求
- 先给出明确结论，再展开要件/情形分析
- 法条引用嵌入分析句子内部（不要单列法条清单）
- 覆盖完整：主要规定、司法解释细化、例外情形、程序衔接
- 内容全面，宁多勿少，确保重要司法解释都被列出
- 适合法考备考场景，条理清晰
- 所有条号必须来自工具检索结果，不得凭记忆填写
"""

_FORCE_FINAL_PROMPT = """\
请根据以上所有工具检索结果，给出完整、准确的最终答案。

严格要求：
1. 所有法条引用必须用 [[law_id|法律名称|条号|标注]] 格式，law_id 从工具结果的 law_id= 字段直接提取
2. 凡是提到法律条文，无论核心还是顺带，都必须用 [[...]] 格式，禁止裸文本引用
3. 区分：主要依据（核心法条）/ 司法解释细化 / 例外情形 / 程序衔接
4. 先给结论，后展开分析，覆盖所有重要法条和司法解释
5. 内容宁多勿少，确保法考复习所需的完整体系都被呈现
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
    ) -> dict:
        """Send a chat completion request; return the full response dict."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = self._session.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=REQUEST_TIMEOUT,
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
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_call_log: list[dict] = []
        rounds = 0

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
                if verbose:
                    logger.info("Agent finished in %d round(s).", rounds)
                return {
                    "answer": answer,
                    "rounds": rounds,
                    "tool_calls": tool_call_log,
                    "messages": messages,
                }

            # ---- Execute each tool call ----
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args = tc["function"].get("arguments", "{}")
                tool_call_id = tc["id"]

                if verbose:
                    logger.info("  → %s(%s)", tool_name, tool_args[:120])

                output = self._execute_tool(tool_name, tool_args)

                tool_call_log.append(
                    {
                        "round": rounds,
                        "tool": tool_name,
                        "args": tool_args,
                        "output_preview": output[:300],
                    }
                )

                # Append tool result to history
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": output,
                    }
                )

        # ---- MAX_ROUNDS reached: force a final synthesis ----
        logger.warning("MAX_ROUNDS (%d) reached; forcing final synthesis.", MAX_ROUNDS)
        messages.append({"role": "user", "content": _FORCE_FINAL_PROMPT})
        response = self._chat(messages, tools=None)  # no tools → must answer
        answer = response["choices"][0]["message"].get("content") or ""
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
