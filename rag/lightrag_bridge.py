from __future__ import annotations

"""
LightRAG bridge for the legal article corpus.

Event loop strategy
-------------------
LightRAG's internal async workers bind themselves to the event loop that is
running when they are first initialised.  Calling asyncio.run() more than
once in the same process creates a *new* loop each time, which triggers
"bound to a different event loop" errors on subsequent queries.

Fix: one persistent daemon thread keeps a single event loop alive for the
entire process lifetime.  All async helpers are scheduled onto that loop via
asyncio.run_coroutine_threadsafe().  The public synchronous wrappers (query,
ingest) use _run_async() which submits work to that persistent loop.

Responsibilities
----------------
1. Initialise a LightRAG instance wired to DeepSeek (LLM) and
   SiliconFlow Qwen3-Embedding-4B (embeddings).
2. Convert the existing law-article corpus + citation graph into
   LightRAG's custom_kg format and insert it (one-time offline step).
3. Expose a thin synchronous query() wrapper used by the Agent tools.

Storage layout
--------------
All LightRAG artefacts live under  rag/storage_lightrag/  so they never
conflict with the existing Qdrant-backed vector index.
"""

import asyncio
import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent background event loop
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return the persistent background event loop, creating it if necessary."""
    global _loop
    if _loop is not None and _loop.is_running():
        return _loop
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        loop = asyncio.new_event_loop()
        _loop = loop

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_run, daemon=True, name="lightrag-event-loop")
        t.start()
        # Give the thread a moment to enter run_forever()
        import time

        time.sleep(0.05)
        return loop


def _run_async(coro) -> Any:
    """
    Submit a coroutine to the persistent background loop and block until done.

    This is the synchronous entry-point for all LightRAG operations.
    """
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=300)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _lightrag_imports():
    from lightrag import LightRAG, QueryParam  # noqa: F401
    from lightrag.utils import EmbeddingFunc  # noqa: F401

    return LightRAG, QueryParam, EmbeddingFunc


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------


def _working_dir() -> str:
    here = Path(__file__).resolve().parent
    d = here / "storage_lightrag"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


EMBED_DIM = 2560  # Qwen/Qwen3-Embedding-4B
EMBED_MAX_TOKENS = 512
EMBED_BATCH = 32  # texts per SiliconFlow request


# ---------------------------------------------------------------------------
# LLM function — DeepSeek via OpenAI-compatible API
# ---------------------------------------------------------------------------


async def _deepseek_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs,
) -> str:
    from rag.config import settings

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in history_messages or []:
        messages.append(msg)
    messages.append({"role": "user", "content": prompt})

    loop = asyncio.get_event_loop()

    def _call() -> str:
        resp = requests.post(
            f"{settings.chat_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.chat_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.chat_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return await loop.run_in_executor(None, _call)


# ---------------------------------------------------------------------------
# Embedding function — SiliconFlow Qwen3-Embedding-4B
# ---------------------------------------------------------------------------


async def _siliconflow_embed(texts: list[str]) -> np.ndarray:
    from rag.config import settings

    loop = asyncio.get_event_loop()

    def _call_batch_with_retry(
        batch: list[str], max_retries: int = 5
    ) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    f"{settings.siliconflow_base_url.rstrip('/')}/embeddings",
                    headers={
                        "Authorization": f"Bearer {settings.siliconflow_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.siliconflow_embedding_model,
                        "input": batch,
                        "encoding_format": "float",
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda x: x["index"])
                return [item["embedding"] for item in data]
            except Exception as exc:
                last_exc = exc
                wait = 2.0 * attempt
                logger.warning(
                    "Embedding batch failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                import time as _time

                _time.sleep(wait)
        raise RuntimeError(
            f"Embedding failed after {max_retries} retries"
        ) from last_exc

    def _call_all() -> np.ndarray:
        results: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            results.extend(_call_batch_with_retry(texts[i : i + EMBED_BATCH]))
        return np.array(results, dtype=np.float32)

    return await loop.run_in_executor(None, _call_all)


# ---------------------------------------------------------------------------
# LightRAG singleton (initialised on the persistent loop)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_lightrag():
    """
    Initialise and cache the LightRAG instance.

    The instance is created inside _run_async so that all of its internal
    async workers are bound to the persistent background event loop.
    """

    async def _make():
        LightRAG, _QueryParam, EmbeddingFunc = _lightrag_imports()

        embed_func = EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=EMBED_MAX_TOKENS,
            func=_siliconflow_embed,
        )

        rag = LightRAG(
            working_dir=_working_dir(),
            llm_model_func=_deepseek_complete,
            embedding_func=embed_func,
            addon_params={
                "language": "Simplified Chinese",
                "entity_types": ["law_article", "law", "concept"],
            },
        )
        await rag.initialize_storages()
        return rag

    return _run_async(_make())


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------


def _build_custom_kg(
    limit: int | None = None,
    law_ids: set[str] | None = None,
) -> dict:
    """
    Convert the law article corpus into LightRAG's custom_kg format.

    Entity   = one law article  (name = "<law_name><article_num>")
    Relation = one citation edge between two articles
    Chunk    = article full text (for vector retrieval in mix/naive modes)
    """
    from rag.loader import build_article_corpus

    articles = build_article_corpus(limit=limit, law_ids=law_ids)
    logger.info("Building custom_kg from %d articles ...", len(articles))

    entities: list[dict] = []
    relationships: list[dict] = []
    chunks: list[dict] = []
    seen_entity_names: set[str] = set()
    seen_rel_keys: set[tuple] = set()

    for article in articles:
        entity_name = f"{article.law_name}{article.article_num}"
        if entity_name in seen_entity_names:
            continue
        seen_entity_names.add(entity_name)

        description_parts: list[str] = []
        if article.annotation:
            description_parts.append(f"【考点】{article.annotation}")
        if article.chapter:
            description_parts.append(f"【章节】{article.chapter}")
        description_parts.append(article.article_text[:600])
        description = "\n".join(description_parts)

        entities.append(
            {
                "entity_name": entity_name,
                "entity_type": "law_article",
                "description": description,
                "source_id": article.business_id,
                "file_path": f"{article.law_id}.json",
            }
        )

        chunks.append(
            {
                "content": article.embedding_text(),
                "source_id": article.business_id,
                "file_path": f"{article.law_id}.json",
            }
        )

        for edge in article.outgoing_citations:
            if not edge.target_law_name or not edge.target_article:
                continue
            tgt_name = f"{edge.target_law_name}{edge.target_article}"
            rel_key = (entity_name, tgt_name)
            if rel_key in seen_rel_keys:
                continue
            seen_rel_keys.add(rel_key)
            ctx = (edge.context or "")[:200].strip()
            rel_desc = f"引用：{ctx}" if ctx else "引用"
            relationships.append(
                {
                    "src_id": entity_name,
                    "tgt_id": tgt_name,
                    "description": rel_desc,
                    "keywords": "引用 参照适用 依照",
                    "weight": 1.0,
                    "source_id": article.business_id,
                    "file_path": f"{article.law_id}.json",
                }
            )

    logger.info(
        "custom_kg ready: %d entities, %d relationships, %d chunks",
        len(entities),
        len(relationships),
        len(chunks),
    )
    return {"entities": entities, "relationships": relationships, "chunks": chunks}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _split_kg(kg: dict, batch_size: int) -> list[dict]:
    """Split a custom_kg dict into smaller batches of at most *batch_size* entities."""
    entities = kg["entities"]
    relationships = kg["relationships"]
    chunks = kg["chunks"]
    batches: list[dict] = []
    for start in range(0, max(len(entities), 1), batch_size):
        end = start + batch_size
        # Collect the entity names in this batch for filtering relations/chunks
        batch_names = {e["entity_name"] for e in entities[start:end]}
        batches.append(
            {
                "entities": entities[start:end],
                # Only include relations where both endpoints are in this batch
                # (avoids referencing entities not yet inserted)
                "relationships": [
                    r
                    for r in relationships
                    if r["src_id"] in batch_names or r["tgt_id"] in batch_names
                ],
                "chunks": chunks[start:end],
            }
        )
    return batches


def ingest(
    limit: int | None = None,
    law_ids: set[str] | None = None,
    batch_size: int = 500,
) -> dict[str, int]:
    """
    Build the custom_kg from the law corpus and insert it into LightRAG.

    This is the offline ingestion step.  Run it once (or after corpus
    updates) before using the query interface.

    The corpus is processed in batches of *batch_size* articles to avoid
    overwhelming the embedding API with a single massive request.  Each
    batch is retried independently on network errors.
    """
    rag = get_lightrag()
    kg = _build_custom_kg(limit=limit, law_ids=law_ids)
    batches = _split_kg(kg, batch_size)
    total = len(kg["entities"])
    logger.info(
        "Inserting %d entities in %d batches (batch_size=%d) ...",
        total,
        len(batches),
        batch_size,
    )

    inserted = 0
    for i, batch in enumerate(batches, 1):
        batch_len = len(batch["entities"])
        logger.info(
            "Batch %d/%d — %d entities (%d done / %d total)",
            i,
            len(batches),
            batch_len,
            inserted,
            total,
        )

        max_batch_retries = 3
        for attempt in range(1, max_batch_retries + 1):
            try:

                async def _do(b=batch) -> None:
                    await rag.ainsert_custom_kg(b)

                _run_async(_do())
                inserted += batch_len
                break
            except Exception as exc:
                if attempt == max_batch_retries:
                    logger.error(
                        "Batch %d failed after %d attempts: %s — skipping.",
                        i,
                        max_batch_retries,
                        exc,
                    )
                    break
                wait = 5.0 * attempt
                logger.warning(
                    "Batch %d attempt %d/%d failed: %s — retrying in %.0fs",
                    i,
                    attempt,
                    max_batch_retries,
                    exc,
                    wait,
                )
                import time as _time

                _time.sleep(wait)

    logger.info(
        "LightRAG ingestion complete. %d / %d entities inserted.", inserted, total
    )
    return {
        "entities": len(kg["entities"]),
        "relationships": len(kg["relationships"]),
        "chunks": len(kg["chunks"]),
        "inserted": inserted,
    }


def query(
    question: str,
    mode: str = "mix",
    top_k: int = 10,
) -> str:
    """
    Query the LightRAG knowledge graph.

    Modes
    -----
    local   -- entity-centric: relevant article nodes + neighbours
    global  -- community summaries: broad overview questions
    hybrid  -- local + global combined
    mix     -- knowledge graph + vector chunks (recommended default)
    naive   -- plain vector search (baseline)
    """
    from lightrag import QueryParam  # type: ignore

    rag = get_lightrag()
    param = QueryParam(mode=mode, top_k=top_k, stream=False)

    async def _do() -> str:
        result = await rag.aquery(question, param=param)
        return str(result)

    return _run_async(_do())


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="LightRAG bridge CLI")
    sub = parser.add_subparsers(dest="cmd")

    ing = sub.add_parser("ingest", help="Ingest law corpus into LightRAG")
    ing.add_argument("--limit", type=int, default=None)
    ing.add_argument(
        "--law-ids",
        type=str,
        default="",
        help="Comma-separated law IDs (e.g. 3346,250)",
    )
    ing.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Articles per ingestion batch (default 500)",
    )

    qry = sub.add_parser("query", help="Query LightRAG")
    qry.add_argument("question")
    qry.add_argument(
        "--mode", default="mix", choices=["local", "global", "hybrid", "mix", "naive"]
    )
    qry.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    if args.cmd == "ingest":
        ids = (
            {s.strip() for s in args.law_ids.split(",") if s.strip()}
            if args.law_ids
            else None
        )
        summary = ingest(limit=args.limit, law_ids=ids, batch_size=args.batch_size)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    elif args.cmd == "query":
        answer = query(args.question, mode=args.mode, top_k=args.top_k)
        print(answer)

    else:
        parser.print_help()
