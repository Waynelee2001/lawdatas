from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """
    Runtime configuration, loaded once from environment variables.

    Provider split
    --------------
    SiliconFlow  – embeddings (Qwen3-Embedding-4B) and reranking
    DeepSeek     – LLM tasks: query analysis + evidence compression
                   Falls back to the SiliconFlow chat model when
                   DEEPSEEK_API_KEY is not set.
    """

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    root_dir: Path = ROOT_DIR
    laws_dir: Path = ROOT_DIR / "data" / "laws_annotation"
    annotations_dir: Path = ROOT_DIR / "data" / "annotations"
    backlinks_path: Path = ROOT_DIR / "js" / "backlinks.js"
    law_map_path: Path = ROOT_DIR / "all_laws_map.json"

    def __init__(self) -> None:
        # Paths
        self.root_dir = ROOT_DIR
        self.laws_dir = ROOT_DIR / "data" / "laws_annotation"
        self.annotations_dir = ROOT_DIR / "data" / "annotations"
        self.backlinks_path = ROOT_DIR / "js" / "backlinks.js"
        self.law_map_path = ROOT_DIR / "all_laws_map.json"
        self.storage_dir = ROOT_DIR / os.getenv("INDEX_STORAGE_DIR", "rag/storage")

        # SiliconFlow — embeddings + reranking
        self.siliconflow_api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
        self.siliconflow_base_url: str = os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
        )
        self.siliconflow_embedding_model: str = os.getenv(
            "SILICONFLOW_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"
        )
        self.siliconflow_rerank_model: str = os.getenv(
            "SILICONFLOW_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B"
        )
        # Fallback chat model on SiliconFlow (used only when DeepSeek is not configured)
        self.siliconflow_chat_model: str = os.getenv(
            "SILICONFLOW_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct"
        )
        self.siliconflow_enable_rerank: bool = (
            os.getenv("SILICONFLOW_ENABLE_RERANK", "1") == "1"
        )
        self.siliconflow_embed_batch_size: int = int(
            os.getenv("SILICONFLOW_EMBED_BATCH_SIZE", "64")
        )
        self.siliconflow_request_batch_size: int = int(
            os.getenv("SILICONFLOW_REQUEST_BATCH_SIZE", "16")
        )
        self.siliconflow_max_parallel_requests: int = int(
            os.getenv("SILICONFLOW_MAX_PARALLEL_REQUESTS", "4")
        )

        # DeepSeek — preferred provider for LLM tasks
        self.deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_base_url: str = os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        )
        self.deepseek_chat_model: str = os.getenv(
            "DEEPSEEK_CHAT_MODEL", "deepseek-chat"
        )
        self.deepseek_query_analysis_model: str = os.getenv(
            "DEEPSEEK_QUERY_ANALYSIS_MODEL",
            os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
        )

        # Qdrant
        self.index_persist_every_batches: int = int(
            os.getenv("INDEX_PERSIST_EVERY_BATCHES", "4")
        )
        self.qdrant_url: str = os.getenv("QDRANT_URL", "")
        self.qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
        self.qdrant_collection: str = os.getenv(
            "QDRANT_COLLECTION", "law_articles_qwen4b"
        )
        self.qdrant_local_path: Path = ROOT_DIR / os.getenv(
            "QDRANT_LOCAL_PATH", "rag/storage/qdrant_local"
        )

    # ------------------------------------------------------------------
    # Effective chat provider (read-only derived properties)
    #
    # Use these throughout the codebase instead of checking DeepSeek vs
    # SiliconFlow directly.
    # ------------------------------------------------------------------

    @property
    def chat_api_key(self) -> str:
        """API key for LLM chat tasks (compression + query analysis)."""
        return self.deepseek_api_key or self.siliconflow_api_key

    @property
    def chat_base_url(self) -> str:
        """Base URL for LLM chat tasks."""
        if self.deepseek_api_key:
            return self.deepseek_base_url
        return self.siliconflow_base_url

    @property
    def chat_model(self) -> str:
        """Model name used for evidence compression."""
        if self.deepseek_api_key:
            return self.deepseek_chat_model
        return self.siliconflow_chat_model

    @property
    def query_analysis_model(self) -> str:
        """Model name used for LLM query analysis."""
        if self.deepseek_api_key:
            return self.deepseek_query_analysis_model
        return self.siliconflow_chat_model

    @property
    def chat_provider(self) -> str:
        """Human-readable name of the active chat provider."""
        return "deepseek" if self.deepseek_api_key else "siliconflow"


settings = Settings()
