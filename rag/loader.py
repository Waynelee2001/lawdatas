from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from rag.config import settings
from rag.schema import CitationEdge, LegalArticle

if TYPE_CHECKING:
    from llama_index.core.schema import TextNode


ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百千万零〇○0-9]+条(?:之[一二三四五六七八九十百千万零〇○0-9]+)?)")
CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百千万零〇○0-9]+章)\s*(.*)$")
SECTION_RE = re.compile(r"^(第[一二三四五六七八九十百千万零〇○0-9]+节)\s*(.*)$")
OUTGOING_CITATION_RE = re.compile(
    r"(?:《([^》]+)》|"
    r"(民法典|刑法|公司法|合伙企业法|民事诉讼法|刑事诉讼法|行政诉讼法|行政处罚法|行政许可法|行政强制法|行政复议法|国家赔偿法|公务员法|治安管理处罚法|票据法|证券法|保险法|海商法|信托法|企业破产法|劳动法|劳动合同法|消费者权益保护法|反不正当竞争法|反垄断法|产品质量法|商标法|专利法|著作权法|仲裁法|公证法|律师法|法律援助法|人民陪审员法|监察法|社区矫正法|立法法|宪法))"
    r"(第[一二三四五六七八九十百千万零〇○0-9]+条(?:之[一二三四五六七八九十百千万零〇○0-9]+)?)"
)


def load_law_name_map() -> dict[str, str]:
    return json.loads(settings.law_map_path.read_text(encoding="utf-8"))


def build_law_name_lookup() -> dict[str, tuple[str, str]]:
    law_map = load_law_name_map()
    lookup: dict[str, tuple[str, str]] = {}
    for law_id, law_name in law_map.items():
        candidates = {
            law_name,
            law_name.replace("《", "").replace("》", ""),
        }
        if law_name.startswith("中华人民共和国"):
            candidates.add(law_name.removeprefix("中华人民共和国"))
        for name in list(candidates):
            if name.endswith("（2023修订）"):
                candidates.add(name.removesuffix("（2023修订）"))
        for candidate in candidates:
            lookup.setdefault(candidate, (law_id, law_name))
    return lookup


def load_annotations(law_id: str) -> tuple[dict[str, str], dict[str, str]]:
    path = settings.annotations_dir / f"{law_id}.json"
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = data.get("annotations", data)
    chapter_annotations = data.get("chapter_annotations", {})
    return annotations, chapter_annotations


def load_backlinks() -> dict[str, dict[str, list[dict[str, str]]]]:
    raw = settings.backlinks_path.read_text(encoding="utf-8").strip()
    prefix = "var globalBacklinks = "
    if raw.startswith(prefix):
        raw = raw[len(prefix) :]
    if raw.endswith(";"):
        raw = raw[:-1]
    return json.loads(raw)


def _iter_content_blocks(law_data: dict) -> list[dict]:
    if isinstance(law_data.get("content"), list):
        return law_data["content"]
    if isinstance(law_data.get("data"), list):
        return law_data["data"]
    return []


def _normalize_text(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_outgoing_citations(
    text: str, *, source_law_id: str, source_law_name: str, law_name_lookup: dict[str, tuple[str, str]]
) -> list[CitationEdge]:
    citations: dict[tuple[str, str, str], CitationEdge] = {}
    for match in OUTGOING_CITATION_RE.finditer(text):
        law_name = match.group(1) or match.group(2) or ""
        article_num = match.group(3)
        target_law_id = ""
        target_law_name = law_name
        if law_name in law_name_lookup:
            target_law_id, target_law_name = law_name_lookup[law_name]
        key = (target_law_id or target_law_name, article_num, source_law_id)
        citations[key] = CitationEdge(
            edge_type="outgoing_citation",
            source_law_id=source_law_id,
            source_law_name=source_law_name,
            source_article="",
            target_law_id=target_law_id,
            target_law_name=target_law_name,
            target_article=article_num,
            context=text[:220],
        )
    return list(citations.values())


def parse_law_articles(
    law_id: str,
    backlinks: dict[str, dict[str, list[dict[str, str]]]],
    law_name_lookup: dict[str, tuple[str, str]],
) -> list[LegalArticle]:
    law_path = settings.laws_dir / f"{law_id}.json"
    law_data = json.loads(law_path.read_text(encoding="utf-8"))
    annotations, chapter_annotations = load_annotations(law_id)

    law_name = law_data.get("name", load_law_name_map().get(law_id, law_id))
    articles: list[LegalArticle] = []
    chapter = ""
    section = ""
    current_article_num = ""
    current_article_lines: list[str] = []
    current_block_id = ""

    def flush_article() -> None:
        nonlocal current_article_num, current_article_lines, current_block_id
        if not current_article_num or not current_article_lines:
            return
        article_text = "\n".join(current_article_lines).strip()
        incoming = [
            CitationEdge(
                edge_type="incoming_citation",
                source_law_id=str(item.get("sourceLawId", "")),
                source_law_name=item.get("sourceLawName", ""),
                source_article=item.get("sourceArticle", ""),
                target_law_id=law_id,
                target_law_name=law_name,
                target_article=current_article_num,
                context=item.get("context", ""),
            )
            for item in backlinks.get(law_id, {}).get(current_article_num, [])
        ]
        outgoing = []
        for edge in _extract_outgoing_citations(
            article_text, source_law_id=law_id, source_law_name=law_name, law_name_lookup=law_name_lookup
        ):
            edge.source_article = current_article_num
            outgoing.append(edge)
        articles.append(
            LegalArticle(
                law_id=law_id,
                law_name=law_name,
                article_num=current_article_num,
                article_text=article_text,
                annotation=annotations.get(current_article_num, ""),
                chapter=chapter,
                chapter_annotation=chapter_annotations.get(chapter, ""),
                section=section,
                source_block_id=current_block_id,
                outgoing_citations=outgoing,
                incoming_citations=incoming,
            )
        )
        current_article_num = ""
        current_article_lines = []
        current_block_id = ""

    for block in _iter_content_blocks(law_data):
        block_id = str(block.get("id", ""))
        text = block.get("lawWebContent") or block.get("content") or ""
        for line in _normalize_text(text):
            chapter_match = CHAPTER_RE.match(line)
            if chapter_match:
                chapter = chapter_match.group(1)
            section_match = SECTION_RE.match(line)
            if section_match:
                section = section_match.group(1)
            article_match = ARTICLE_RE.match(line)
            if article_match:
                flush_article()
                current_article_num = article_match.group(1)
                current_block_id = block_id
                current_article_lines = [line]
                continue
            if current_article_num:
                current_article_lines.append(line)

    flush_article()
    return articles


def build_article_corpus(limit: int | None = None, law_ids: set[str] | None = None) -> list[LegalArticle]:
    backlinks = load_backlinks()
    law_name_lookup = build_law_name_lookup()
    law_files = sorted(p for p in settings.laws_dir.glob("*.json"))
    articles: list[LegalArticle] = []
    seen_business_ids: set[str] = set()
    for law_file in law_files:
        law_id = law_file.stem
        if law_ids is not None and law_id not in law_ids:
            continue
        for article in parse_law_articles(law_id, backlinks, law_name_lookup):
            if article.business_id in seen_business_ids:
                continue
            seen_business_ids.add(article.business_id)
            articles.append(article)
        if limit is not None and len(articles) >= limit:
            return articles[:limit]
    return articles


def build_article_lookup(limit: int | None = None, law_ids: set[str] | None = None) -> dict[str, LegalArticle]:
    return {article.business_id: article for article in build_article_corpus(limit=limit, law_ids=law_ids)}


def build_text_nodes(limit: int | None = None, law_ids: set[str] | None = None) -> list["TextNode"]:
    from llama_index.core.schema import TextNode

    nodes: list[TextNode] = []
    for article in build_article_corpus(limit=limit, law_ids=law_ids):
        nodes.append(
            TextNode(
                id_=article.node_id,
                text=article.embedding_text(),
                metadata=article.metadata(),
            )
        )
    return nodes
