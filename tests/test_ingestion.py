"""Tests for the ingestion pipeline: chunking, loading and extraction."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.core.ingestion.chunker import TextChunker
from app.core.ingestion.document_loader import (
    DocumentLoader,
    UnsupportedFileTypeError,
)
from app.core.ingestion.entity_extractor import EntityExtractor


# ----------------------------------------------------------------------
# Chunker
# ----------------------------------------------------------------------
def test_chunker_respects_sentences() -> None:
    """Chunk boundaries fall between sentences, never inside one."""
    text = " ".join(
        f"Sentence number {i} has some filler words here." for i in range(20)
    )
    pages = [{"text": text, "page_num": 1, "file_name": "d.txt"}]
    chunks = TextChunker(chunk_size=60, overlap=0).chunk(pages)  # ~45 words/chunk

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["block_type"] == "text"
        # Every text chunk ends at a sentence boundary.
        assert chunk["text"].rstrip().endswith(".")
    # Every original sentence survives intact somewhere.
    joined = " ".join(chunk["text"] for chunk in chunks)
    for i in range(20):
        assert f"Sentence number {i} has some filler words here." in joined


def test_chunker_sentence_overlap() -> None:
    """Overlap is carried as whole trailing sentences, never a partial word."""
    text = " ".join(f"This is sentence {i} of the document." for i in range(30))
    pages = [{"text": text, "page_num": 1, "file_name": "d.txt"}]
    chunks = TextChunker(chunk_size=60, overlap=20).chunk(pages)

    assert len(chunks) > 1
    # The overlap between two chunks is a whole shared sentence, so no chunk
    # begins or ends with a fragment (all boundaries land on a full stop).
    for chunk in chunks:
        assert chunk["text"].rstrip().endswith(".")


def test_chunker_keeps_table_atomic() -> None:
    """A markdown table becomes one dedicated chunk, not merged with prose."""
    md = (
        "Intro paragraph here.\n\n"
        "| Name | Role |\n| --- | --- |\n| Alice | Dev |\n| Bob | PM |\n\n"
        "Closing paragraph."
    )
    pages = [{"text": md, "page_num": 1, "file_name": "t.md"}]
    chunks = TextChunker(chunk_size=512, overlap=0).chunk(pages)

    tables = [c for c in chunks if c["block_type"] == "table"]
    assert len(tables) == 1
    assert "| Alice | Dev |" in tables[0]["text"]
    assert "| Name | Role |" in tables[0]["text"]
    # The table is not contaminated with surrounding prose.
    assert "Intro paragraph" not in tables[0]["text"]
    assert "Closing paragraph" not in tables[0]["text"]


def test_chunker_separates_images() -> None:
    """A markdown image reference becomes its own image chunk."""
    md = "Some text.\n\n![a diagram](img/diagram.png)\n\nMore text."
    pages = [{"text": md, "page_num": 1, "file_name": "i.md"}]
    chunks = TextChunker(chunk_size=512, overlap=0).chunk(pages)

    images = [c for c in chunks if c["block_type"] == "image"]
    assert len(images) == 1
    assert "diagram.png" in images[0]["text"]


def test_chunker_tracks_section() -> None:
    """Chunks record the heading of the section they belong to."""
    md = (
        "# Introduction\n\nGraphRAG combines vectors and graphs.\n\n"
        "## Method\n\nWe chunk semantically."
    )
    pages = [{"text": md, "page_num": 1, "file_name": "h.md"}]
    chunks = TextChunker(chunk_size=512, overlap=0).chunk(pages)

    intro = next(c for c in chunks if "combines vectors" in c["text"])
    assert intro["section"] == "Introduction"
    method = next(c for c in chunks if "chunk semantically" in c["text"])
    assert method["section"] == "Method"


def test_chunker_short_text() -> None:
    """Text shorter than the chunk size yields a single chunk."""
    pages = [{"text": "only a few words here", "page_num": 1, "file_name": "s.txt"}]
    chunks = TextChunker(chunk_size=512, overlap=50).chunk(pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "only a few words here"
    assert chunks[0]["block_type"] == "text"
    assert chunks[0]["chunk_index"] == 0


def test_chunker_long_runon_fallback() -> None:
    """A single over-long run-on (no punctuation) is hard-split by words."""
    words = " ".join(f"word{i}" for i in range(1000))
    pages = [{"text": words, "page_num": 1, "file_name": "doc.txt"}]
    chunks = TextChunker(chunk_size=100, overlap=10).chunk(pages)

    assert len(chunks) > 1
    assert len(chunks[0]["text"].split()) == int(100 * 0.75)  # 75 words
    doc_id = chunks[0]["doc_id"]
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i
        assert chunk["chunk_id"] == f"{doc_id}_{i}"
        assert chunk["block_type"] == "text"


# ----------------------------------------------------------------------
# DocumentLoader
# ----------------------------------------------------------------------
def test_document_loader_txt(tmp_path) -> None:
    """A .txt file is loaded into a single page record."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello GraphRAG world.", encoding="utf-8")
    loader = DocumentLoader()
    pages = loader.load(str(file_path))
    assert len(pages) == 1
    assert pages[0]["text"] == "Hello GraphRAG world."
    assert pages[0]["file_name"] == "sample.txt"
    assert pages[0]["page_num"] == 1


def test_document_loader_md_from_bytes() -> None:
    """A markdown document loads from raw bytes."""
    loader = DocumentLoader()
    pages = loader.load_from_bytes(b"# Title\n\nBody text", "notes.md")
    assert len(pages) == 1
    assert "Title" in pages[0]["text"]


def test_document_loader_unsupported() -> None:
    """An unsupported extension raises UnsupportedFileTypeError."""
    loader = DocumentLoader()
    with pytest.raises(UnsupportedFileTypeError):
        loader.load_from_bytes(b"data", "spreadsheet.xlsx")


# ----------------------------------------------------------------------
# EntityExtractor
# ----------------------------------------------------------------------
def _make_extractor() -> EntityExtractor:
    """Build an EntityExtractor without touching the real Gemini client."""
    with patch("app.core.ingestion.entity_extractor.genai") as mock_genai:
        mock_genai.GenerativeModel.return_value = MagicMock()
        extractor = EntityExtractor.__new__(EntityExtractor)
        extractor._settings = MagicMock()  # type: ignore[attr-defined]
        extractor._model = MagicMock()  # type: ignore[attr-defined]
    return extractor


def test_entity_extractor_parse() -> None:
    """A well-formed JSON response is parsed into entities/relationships."""
    extractor = _make_extractor()
    raw = (
        '{"entities": [{"name": "Neo4j", "type": "TECHNOLOGY", '
        '"description": "A graph database."}], '
        '"relationships": [{"source": "Neo4j", "relation": "IS_A", '
        '"target": "Database", "description": "Neo4j is a database."}]}'
    )
    extractor._model.generate_content.return_value = MagicMock(text=raw)  # type: ignore[attr-defined]
    result = extractor.extract("Neo4j is a graph database.")
    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == "Neo4j"
    assert len(result["relationships"]) == 1


def test_entity_extractor_strips_code_fence() -> None:
    """JSON wrapped in a markdown code fence is still parsed."""
    extractor = _make_extractor()
    raw = '```json\n{"entities": [], "relationships": []}\n```'
    extractor._model.generate_content.return_value = MagicMock(text=raw)  # type: ignore[attr-defined]
    result = extractor.extract("some text")
    assert result == {"entities": [], "relationships": []}


def test_entity_extractor_malformed_json() -> None:
    """Garbage model output yields an empty structure without raising."""
    extractor = _make_extractor()
    extractor._model.generate_content.return_value = MagicMock(  # type: ignore[attr-defined]
        text="this is not json at all"
    )
    result = extractor.extract("some text")
    assert result == {"entities": [], "relationships": []}


# ----------------------------------------------------------------------
# GraphBuilder duplicate detection
# ----------------------------------------------------------------------
def test_duplicate_detection() -> None:
    """A document already marked ingested is skipped on the second call."""
    from app.core.graph.graph_builder import GraphBuilder

    client = MagicMock()
    client.document_exists.return_value = True
    builder = GraphBuilder(client, MagicMock(), MagicMock())

    chunks = [
        {
            "chunk_id": "abc_0",
            "text": "hello",
            "doc_id": "abc",
            "chunk_index": 0,
            "page_num": 1,
        }
    ]
    summary = asyncio.run(builder.build_from_chunks(chunks))
    assert summary["chunks_processed"] == 0
    client.upsert_chunk.assert_not_called()
