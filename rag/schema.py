from __future__ import annotations

from dataclasses import dataclass, field
from uuid import NAMESPACE_URL, uuid5


@dataclass
class CitationEdge:
    edge_type: str
    source_law_id: str
    source_law_name: str
    source_article: str
    target_law_id: str
    target_law_name: str
    target_article: str
    context: str = ""

    @property
    def source_business_id(self) -> str:
        return f"{self.source_law_id}:{self.source_article}"

    @property
    def target_business_id(self) -> str:
        if not self.target_law_id or not self.target_article:
            return ""
        return f"{self.target_law_id}:{self.target_article}"


@dataclass
class LegalArticle:
    law_id: str
    law_name: str
    article_num: str
    article_text: str
    annotation: str = ""
    chapter: str = ""
    chapter_annotation: str = ""
    section: str = ""
    source_block_id: str = ""
    outgoing_citations: list[CitationEdge] = field(default_factory=list)
    incoming_citations: list[CitationEdge] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return str(uuid5(NAMESPACE_URL, self.business_id))

    @property
    def business_id(self) -> str:
        return f"{self.law_id}:{self.article_num}"

    def embedding_text(self) -> str:
        lines = [
            f"法律名称：{self.law_name}",
            f"条号：{self.article_num}",
        ]
        if self.chapter:
            lines.append(f"章节：{self.chapter}")
        if self.chapter_annotation:
            lines.append(f"章节标注：{self.chapter_annotation}")
        if self.annotation:
            lines.append(f"标注：{self.annotation}")
        lines.append(f"正文：{self.article_text}")
        return "\n".join(lines)

    def metadata(self) -> dict[str, object]:
        return {
            "business_id": self.business_id,
            "law_id": self.law_id,
            "law_name": self.law_name,
            "article_num": self.article_num,
            "annotation": self.annotation,
            "chapter": self.chapter,
            "chapter_annotation": self.chapter_annotation,
            "section": self.section,
            "source_block_id": self.source_block_id,
            "outgoing_citation_count": len(self.outgoing_citations),
            "outgoing_business_ids": [
                edge.target_business_id for edge in self.outgoing_citations if edge.target_business_id
            ],
            "outgoing_citations": [
                {
                    "edge_type": edge.edge_type,
                    "target_law_id": edge.target_law_id,
                    "target_law_name": edge.target_law_name,
                    "target_article": edge.target_article,
                    "target_business_id": edge.target_business_id,
                    "context": edge.context,
                }
                for edge in self.outgoing_citations
            ],
            "incoming_citation_count": len(self.incoming_citations),
            "incoming_business_ids": [
                edge.source_business_id for edge in self.incoming_citations if edge.source_business_id
            ],
            "incoming_citations": [
                {
                    "edge_type": edge.edge_type,
                    "source_law_id": edge.source_law_id,
                    "source_law_name": edge.source_law_name,
                    "source_article": edge.source_article,
                    "source_business_id": edge.source_business_id,
                    "context": edge.context,
                }
                for edge in self.incoming_citations
            ],
        }
