from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.config import settings


def _collection_exists(client: QdrantClient, collection: str) -> bool:
    try:
        client.get_collection(collection)
        return True
    except Exception:
        return False


def _extract_vector_config(info: Any) -> Any:
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        return vectors
    return models.VectorParams(size=vectors.size, distance=vectors.distance)


def migrate_collection(
    *,
    source_path: Path,
    source_collection: str,
    target_url: str,
    target_api_key: str,
    target_collection: str,
    batch_size: int,
    recreate: bool,
    skip_first: int,
    max_retries: int,
    timeout: float,
) -> int:
    source = QdrantClient(path=str(source_path))
    target = QdrantClient(url=target_url, api_key=target_api_key or None, timeout=timeout)

    source_info = source.get_collection(source_collection)
    vectors_config = _extract_vector_config(source_info)

    if recreate and _collection_exists(target, target_collection):
        target.delete_collection(target_collection)

    if not _collection_exists(target, target_collection):
        target.create_collection(
            collection_name=target_collection,
            vectors_config=vectors_config,
        )

    offset = None
    migrated = 0
    skipped = 0
    while True:
        points, offset = source.scroll(
            collection_name=source_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        if skipped < skip_first:
            remaining_skip = skip_first - skipped
            if remaining_skip >= len(points):
                skipped += len(points)
                if offset is None:
                    break
                continue
            points = points[remaining_skip:]
            skipped = skip_first

        upsert_points = [
            models.PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload or {},
            )
            for point in points
        ]
        for attempt in range(1, max_retries + 1):
            try:
                target.upsert(
                    collection_name=target_collection,
                    points=upsert_points,
                    wait=True,
                )
                break
            except Exception:
                if attempt >= max_retries:
                    raise
                wait = min(30, attempt * 3)
                print(
                    {"stage": "retry_upsert", "attempt": attempt, "wait": wait},
                    flush=True,
                )
                time.sleep(wait)
        migrated += len(points)
        print(
            {"stage": "migrate_batch", "migrated": migrated, "skipped": skipped},
            flush=True,
        )

        if offset is None:
            break

    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate the local Qdrant collection to Qdrant Cloud."
    )
    parser.add_argument(
        "--source-path",
        default=str(settings.root_dir / "rag" / "storage_full" / "qdrant_local"),
        help="Local Qdrant path.",
    )
    parser.add_argument(
        "--source-collection",
        default="law_articles_qwen4b_full",
        help="Local collection name.",
    )
    parser.add_argument(
        "--target-url",
        default=os.getenv("QDRANT_URL", ""),
        help="Qdrant Cloud URL.",
    )
    parser.add_argument(
        "--target-api-key",
        default=os.getenv("QDRANT_API_KEY", ""),
        help="Qdrant Cloud API key.",
    )
    parser.add_argument(
        "--target-collection",
        default=os.getenv("QDRANT_COLLECTION", "law_articles_qwen4b_full"),
        help="Qdrant Cloud collection name.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip the first N local points before uploading. Useful for resuming.",
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    if not args.target_url:
        raise SystemExit("QDRANT_URL is required.")

    migrated = migrate_collection(
        source_path=Path(args.source_path),
        source_collection=args.source_collection,
        target_url=args.target_url,
        target_api_key=args.target_api_key,
        target_collection=args.target_collection,
        batch_size=max(1, args.batch_size),
        recreate=args.recreate,
        skip_first=max(0, args.skip_first),
        max_retries=max(1, args.max_retries),
        timeout=max(1.0, args.timeout),
    )
    print({"stage": "done", "migrated": migrated})


if __name__ == "__main__":
    main()
