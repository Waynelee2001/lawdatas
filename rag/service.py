from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag.compression import (
    SiliconFlowChatCompressor,
    fallback_evidence,
    is_listing_query,
)
from rag.config import settings
from rag.query_analysis import analyze_query, analyze_query_with_llm
from rag.retrieval import BM25ArticleIndex, RetrievedArticle, graph_retrieve
from rag.topic_grouping import group_topic_candidates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------


def _resolve_local_collection() -> tuple[Path, str]:
    primary_path = settings.qdrant_local_path
    primary_collection = settings.qdrant_collection
    if (primary_path / "collection" / primary_collection).exists():
        return primary_path, primary_collection

    fallbacks = [
        (
            settings.root_dir / "rag" / "storage_full" / "qdrant_local",
            "law_articles_qwen4b_full",
        ),
        (
            settings.root_dir / "rag" / "storage_core2" / "qdrant_local",
            "law_articles_qwen4b_core2",
        ),
        (
            settings.root_dir / "rag" / "storage_parallel_sample" / "qdrant_local",
            "law_articles_qwen4b_parallel_sample",
        ),
    ]
    for path, collection in fallbacks:
        if (path / "collection" / collection).exists():
            return path, collection
    return primary_path, primary_collection


def _build_vector_store():
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    if settings.qdrant_url:
        from rag.index import build_qdrant_client

        client = build_qdrant_client()
        return QdrantVectorStore(
            client=client, collection_name=settings.qdrant_collection
        )

    local_path, collection = _resolve_local_collection()
    local_path.mkdir(parents=True, exist_ok=True)
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(local_path))
    return QdrantVectorStore(client=client, collection_name=collection)


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embed_model():
    from rag.siliconflow import SiliconFlowEmbedding

    return SiliconFlowEmbedding(
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
        model_name=settings.siliconflow_embedding_model,
        embed_batch_size=settings.siliconflow_embed_batch_size,
        request_batch_size=settings.siliconflow_request_batch_size,
        max_parallel_requests=settings.siliconflow_max_parallel_requests,
    )


@lru_cache(maxsize=1)
def get_index():
    from llama_index.core import Settings, VectorStoreIndex

    embed_model = get_embed_model()
    Settings.embed_model = embed_model
    vector_store = _build_vector_store()
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )


@lru_cache(maxsize=1)
def get_reranker():
    if not settings.siliconflow_enable_rerank or not settings.siliconflow_api_key:
        return None
    from rag.siliconflow_rerank import SiliconFlowReranker

    return SiliconFlowReranker(
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
        model_name=settings.siliconflow_rerank_model,
    )


@lru_cache(maxsize=1)
def get_compressor() -> SiliconFlowChatCompressor | None:
    if not settings.chat_api_key:
        return None
    return SiliconFlowChatCompressor(
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        model_name=settings.chat_model,
    )


@lru_cache(maxsize=8)
def get_bm25_index(law_ids_frozen: frozenset[str] | None = None) -> BM25ArticleIndex:
    """
    Build and cache a BM25ArticleIndex.

    The index is keyed by the frozenset of law IDs so that subset queries
    (e.g. searching only within a handful of laws) get their own compact
    index, while the full-corpus index (key=None) is built once and reused.

    Building the full-corpus index takes a few seconds on first call; all
    subsequent calls return the cached object instantly.
    """
    from rag.loader import build_article_corpus

    law_ids = set(law_ids_frozen) if law_ids_frozen else None
    logger.info(
        "Building BM25 index (law_ids=%s) …",
        sorted(law_ids) if law_ids else "full corpus",
    )
    articles = build_article_corpus(law_ids=law_ids)
    return BM25ArticleIndex(articles)


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def serialize_result(result: RetrievedArticle) -> dict[str, Any]:
    article = result.article
    return {
        "law_id": article.law_id,
        "law_name": article.law_name,
        "article_num": article.article_num,
        "annotation": article.annotation,
        "chapter": article.chapter,
        "chapter_annotation": article.chapter_annotation,
        "article_text": article.article_text,
        "business_id": article.business_id,
        "score": result.score,
        "semantic_rerank_score": result.semantic_rerank_score,
        "rerank_score": result.rerank_score,
        "reasons": sorted(result.reasons),
        "matched_terms": result.matched_terms,
        "incoming_citations": [
            {
                "source_law_id": edge.source_law_id,
                "source_law_name": edge.source_law_name,
                "source_article": edge.source_article,
                "context": edge.context,
            }
            for edge in article.incoming_citations
        ],
        "outgoing_citations": [
            {
                "target_law_id": edge.target_law_id,
                "target_law_name": edge.target_law_name,
                "target_article": edge.target_article,
                "context": edge.context,
            }
            for edge in article.outgoing_citations
        ],
    }


# ---------------------------------------------------------------------------
# Main query entry point
# ---------------------------------------------------------------------------


def run_rag_query(
    query: str,
    *,
    top_k: int = 6,
    graph_expand_k: int = 3,
    law_ids: set[str] | None = None,
    compress: bool = True,
) -> dict[str, Any]:
    """
    Execute the full hybrid RAG pipeline and return a serialisable result dict.

    Pipeline
    --------
    1. LLM query analysis — extracts concept, expanded terms, preferred laws,
       scenario terms, and anchor article references for *any* legal question.
       Falls back to rule-based analysis when the LLM call fails or no API
       key is configured.
    2. Hybrid retrieval — dense vector search merged with BM25 sparse search
       via Reciprocal Rank Fusion, followed by one-hop citation graph
       expansion and two-stage reranking.
    3. LLM evidence compression — structures the top candidates into
       primary_basis / supporting_basis / citation_paths / summary.
    4. Topic grouping — organises candidates into base rules, topic-specific
       rules, and weakly related articles for front-end display.
    """

    # ------------------------------------------------------------------ 1 --
    # Query analysis: try LLM first, fall back to rule-based
    query_profile = _analyze_query(query)
    logger.info(
        "Query profile — type=%s concept=%r topic=%r llm=%s provider=%s expanded_terms=%s",
        query_profile.query_type,
        query_profile.concept,
        query_profile.topic,
        query_profile.llm_analyzed,
        settings.chat_provider,
        list(query_profile.expanded_terms)[:6],
    )

    # ------------------------------------------------------------------ 2 --
    # Resolve law_ids cache key (frozenset is hashable → works with lru_cache)
    law_ids_frozen: frozenset[str] | None = frozenset(law_ids) if law_ids else None

    # BM25 index — built lazily on first call, then cached in memory
    bm25_index: BM25ArticleIndex | None = None
    try:
        bm25_index = get_bm25_index(law_ids_frozen)
    except Exception as exc:
        logger.warning(
            "BM25 index unavailable (%s); proceeding with vector-only retrieval.", exc
        )

    results = graph_retrieve(
        get_index(),
        query,
        top_k=top_k,
        graph_expand_k=graph_expand_k,
        law_ids=law_ids,
        reranker=get_reranker(),
        bm25_index=bm25_index,
        query_profile=query_profile,
    )

    # ------------------------------------------------------------------ 3 --
    compressor = get_compressor()
    compress_top_n = min(10 if is_listing_query(query) else 6, len(results))
    if compress and compressor is not None:
        try:
            evidence = compressor.compress(
                query=query, candidates=results, top_n=compress_top_n
            ).to_dict()
        except Exception as exc:
            logger.warning("LLM compression failed (%s); using fallback evidence.", exc)
            evidence = fallback_evidence(query=query, candidates=results).to_dict()
    else:
        evidence = fallback_evidence(query=query, candidates=results).to_dict()

    # ------------------------------------------------------------------ 4 --
    topic_groups = group_topic_candidates(query, results[: max(top_k, 12)])

    return {
        "query": query,
        "query_analysis": {
            "query_type": query_profile.query_type,
            "concept": query_profile.concept,
            "asset_type": getattr(query_profile, "asset_type", ""),
            "topic": query_profile.topic,
            "llm_analyzed": query_profile.llm_analyzed,
            "expanded_terms": list(query_profile.expanded_terms),
            "scenario_terms": list(query_profile.scenario_terms),
            "anchor_refs": [
                {"law_name": anchor.law_name, "article_num": anchor.article_num}
                for anchor in query_profile.anchor_refs
            ],
        },
        "top_k": top_k,
        "graph_expand_k": graph_expand_k,
        "results": [serialize_result(item) for item in results[:top_k]],
        "evidence": evidence,
        "topic_groups": topic_groups,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _analyze_query(query: str):
    """
    Run LLM-powered query analysis when possible; fall back to rule-based.

    Uses DeepSeek when DEEPSEEK_API_KEY is configured in .env, otherwise
    falls back to the SiliconFlow chat model.  If the LLM call fails for
    any reason, rule-based analysis is used as a final safety net.
    """
    if not settings.chat_api_key:
        return analyze_query(query)

    try:
        return analyze_query_with_llm(
            query,
            api_key=settings.chat_api_key,
            base_url=settings.chat_base_url,
            model_name=settings.query_analysis_model,
            timeout=12.0,
        )
    except Exception as exc:
        logger.warning(
            "analyze_query_with_llm raised an unexpected error (%s); using rule-based fallback.",
            exc,
        )
        return analyze_query(query)
