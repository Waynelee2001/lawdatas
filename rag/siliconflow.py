from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import Field, PrivateAttr


class SiliconFlowEmbedding(BaseEmbedding):
    model_name: str = Field(default="Qwen/Qwen3-Embedding-4B")
    api_key: str = Field(default="")
    base_url: str = Field(default="https://api.siliconflow.cn/v1")
    timeout: float = Field(default=120.0)
    max_retries: int = Field(default=4)
    retry_backoff_seconds: float = Field(default=2.0)
    request_batch_size: int = Field(default=16)
    max_parallel_requests: int = Field(default=4)

    _session: requests.Session = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.max_parallel_requests,
            pool_maxsize=self.max_parallel_requests,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    @classmethod
    def class_name(cls) -> str:
        return "SiliconFlowEmbedding"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY is required")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    f"{self.base_url.rstrip('/')}/embeddings",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()["data"]
                data.sort(key=lambda item: item["index"])
                return [item["embedding"] for item in data]
            except (requests.exceptions.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)
        if last_error is None:
            raise RuntimeError("Embedding request failed without a captured exception.")
        raise last_error

    def _embed_chunked_parallel(self, texts: List[str]) -> List[List[float]]:
        if len(texts) <= self.request_batch_size or self.max_parallel_requests <= 1:
            return self._embed_batch(texts)

        batches = [
            (start, texts[start : start + self.request_batch_size])
            for start in range(0, len(texts), self.request_batch_size)
        ]

        def worker(item: tuple[int, List[str]]) -> tuple[int, List[List[float]]]:
            start, chunk = item
            return start, self._embed_batch(chunk)

        ordered_embeddings: list[List[float] | None] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
            for start, embeddings in executor.map(worker, batches):
                for offset, embedding in enumerate(embeddings):
                    ordered_embeddings[start + offset] = embedding

        return [embedding for embedding in ordered_embeddings if embedding is not None]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_batch([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_chunked_parallel(texts)
