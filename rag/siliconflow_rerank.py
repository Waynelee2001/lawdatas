from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class RerankResult:
    index: int
    relevance_score: float


class SiliconFlowReranker:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY is required")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def rerank(self, *, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        if not documents:
            return []
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
            "return_documents": False,
        }
        response = self._session.post(
            f"{self.base_url}/rerank",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json().get("results", [])
        return [
            RerankResult(
                index=int(item["index"]),
                relevance_score=float(item["relevance_score"]),
            )
            for item in data
        ]
