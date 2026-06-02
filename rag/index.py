from __future__ import annotations

import argparse
from math import ceil
from typing import TypeVar

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag.config import settings
from rag.loader import build_text_nodes
from rag.siliconflow import SiliconFlowEmbedding


T = TypeVar("T")


def parse_law_ids(raw: str) -> set[str] | None:
    law_ids = {item.strip() for item in raw.split(",") if item.strip()}
    return law_ids or None


def build_qdrant_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_local_path))


def _iter_batches(items: list[T], batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield idx // batch_size + 1, items[idx : idx + batch_size]


def _persist_index(index: VectorStoreIndex) -> None:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(settings.storage_dir))


def _insert_batch_with_fallback(
    index: VectorStoreIndex, batch: list, *, min_batch_size: int = 64, depth: int = 0
) -> int:
    try:
        index.insert_nodes(batch)
        return 1
    except Exception:
        if len(batch) <= min_batch_size:
            raise
        midpoint = len(batch) // 2
        left = batch[:midpoint]
        right = batch[midpoint:]
        print(
            {
                "stage": "split_batch",
                "depth": depth,
                "original_batch_size": len(batch),
                "left_size": len(left),
                "right_size": len(right),
            }
        )
        return _insert_batch_with_fallback(index, left, min_batch_size=min_batch_size, depth=depth + 1) + _insert_batch_with_fallback(index, right, min_batch_size=min_batch_size, depth=depth + 1)


def build_index(
    limit: int | None = None, *, law_ids: set[str] | None = None, batch_size: int = 256
) -> VectorStoreIndex:
    nodes = build_text_nodes(limit=limit, law_ids=law_ids)
    print({"stage": "load_nodes", "node_count": len(nodes)})
    embed_model = SiliconFlowEmbedding(
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
        model_name=settings.siliconflow_embedding_model,
        embed_batch_size=settings.siliconflow_embed_batch_size,
        request_batch_size=settings.siliconflow_request_batch_size,
        max_parallel_requests=settings.siliconflow_max_parallel_requests,
    )
    client = build_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        embed_model=embed_model,
        insert_batch_size=batch_size,
        show_progress=True,
    )
    total_batches = ceil(len(nodes) / batch_size) if nodes else 0
    inserted = 0
    for batch_no, batch in _iter_batches(nodes, batch_size):
        split_count = _insert_batch_with_fallback(index, batch)
        inserted += len(batch)
        if batch_no % settings.index_persist_every_batches == 0 or inserted == len(nodes):
            _persist_index(index)
        print(
            {
                "stage": "insert_batch",
                "batch": batch_no,
                "total_batches": total_batches,
                "inserted": inserted,
                "node_count": len(nodes),
                "effective_sub_batches": split_count,
                "persisted": batch_no % settings.index_persist_every_batches == 0 or inserted == len(nodes),
            }
        )
    _persist_index(index)
    setattr(index, "_inserted_node_count", len(nodes))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build article-level legal RAG index.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--law-ids", type=str, default="", help="Comma-separated law ids for subset mode.")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of nodes per insert batch.")
    args = parser.parse_args()
    law_ids = parse_law_ids(args.law_ids)
    index = build_index(limit=args.limit, law_ids=law_ids, batch_size=args.batch_size)
    print(
        {
            "collection": settings.qdrant_collection,
            "persist_dir": str(settings.storage_dir),
            "qdrant_mode": settings.qdrant_url or str(settings.qdrant_local_path),
            "node_count": getattr(index, "_inserted_node_count", len(index.index_struct.nodes_dict)),
            "law_ids": sorted(law_ids) if law_ids else None,
            "batch_size": args.batch_size,
        }
    )


if __name__ == "__main__":
    main()
