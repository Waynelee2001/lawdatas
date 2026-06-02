from __future__ import annotations

import argparse

from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from rag.config import settings
from rag.retrieval import graph_retrieve
from rag.siliconflow import SiliconFlowEmbedding
from rag.siliconflow_rerank import SiliconFlowReranker
from rag.index import build_qdrant_client


def load_index() -> VectorStoreIndex:
    embed_model = SiliconFlowEmbedding(
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
        model_name=settings.siliconflow_embedding_model,
        embed_batch_size=settings.siliconflow_embed_batch_size,
        request_batch_size=settings.siliconflow_request_batch_size,
        max_parallel_requests=settings.siliconflow_max_parallel_requests,
    )
    Settings.embed_model = embed_model
    client = build_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the legal RAG index.")
    parser.add_argument("query", help="Question to search against the legal corpus")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--graph-expand-k", type=int, default=2)
    parser.add_argument("--law-ids", type=str, default="", help="Comma-separated law ids for subset mode.")
    args = parser.parse_args()

    index = load_index()
    reranker = None
    if settings.siliconflow_enable_rerank:
        reranker = SiliconFlowReranker(
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
            model_name=settings.siliconflow_rerank_model,
        )
    law_ids = {item.strip() for item in args.law_ids.split(",") if item.strip()} or None
    results = graph_retrieve(
        index,
        args.query,
        top_k=args.top_k,
        graph_expand_k=args.graph_expand_k,
        law_ids=law_ids,
        reranker=reranker,
    )
    for idx, result in enumerate(results[: args.top_k], start=1):
        article = result.article
        print(f"[{idx}] {article.law_name} {article.article_num}")
        if article.annotation:
            print(f"    标注: {article.annotation}")
        print(f"    score: {result.score}")
        print(f"    semantic_rerank_score: {result.semantic_rerank_score}")
        print(f"    rerank_score: {result.rerank_score}")
        print(f"    reasons: {', '.join(sorted(result.reasons))}")
        if result.matched_terms:
            print(f"    matched_terms: {', '.join(result.matched_terms[:8])}")
        text = article.embedding_text().replace("\n", " ")
        print(f"    text: {text[:220]}...")


if __name__ == "__main__":
    main()
