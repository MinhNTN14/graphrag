"""Structure-aware (semantic) text chunking.

Instead of blindly slicing a fixed number of words with overlap — which cuts
through the middle of sentences and destroys meaning — this chunker respects the
*structure* of the document:

1. The text is parsed into **blocks**: headings, paragraphs, tables, images and
   fenced code. Tables, images and code blocks are kept **atomic** (never split
   and never merged with surrounding prose), so a table or figure becomes its
   own chunk.
2. Prose is split into **sentences**; sentences are packed into chunks up to the
   size budget and boundaries always fall **between** sentences, never inside
   one. A single run-on sentence longer than the budget is hard-split by words
   as a last resort.
3. Overlap between adjacent prose chunks is carried at **sentence** granularity
   (whole trailing sentences), so overlap never straddles a word.
4. Each chunk records the **section heading** it falls under and a
   ``block_type`` (``text`` / ``table`` / ``image`` / ``code``).

Token counts are approximated with a word heuristic (1 token ~= 0.75 words).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from loguru import logger

# 1 token is approximately 0.75 words.
_WORDS_PER_TOKEN = 0.75

# A markdown ATX heading: up to three leading spaces then 1-6 '#'.
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
# A markdown image: ![alt](url).
_MD_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)")
# A numbered section heading, e.g. "3 Method" or "3.1 PagedAttention".
_NUM_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z].{0,60}$")
# Split after sentence-ending punctuation followed by whitespace and the start
# of a new sentence (capital letter, digit, or opening quote/bracket).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“'(\[])")

# Standalone section titles common in articles/reports.
_SECTION_WORDS = {
    "abstract", "introduction", "background", "related work", "method",
    "methods", "methodology", "approach", "design", "implementation",
    "evaluation", "experiments", "results", "discussion", "conclusion",
    "conclusions", "references", "acknowledgements", "appendix",
}


class TextChunker:
    """Split documents into semantic, structure-aware chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            overlap: Overlap between adjacent prose chunks, in tokens. Applied at
                sentence granularity (whole trailing sentences up to this budget).
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._max_words = max(1, int(chunk_size * _WORDS_PER_TOKEN))
        self._overlap_words = max(0, int(overlap * _WORDS_PER_TOKEN))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split page documents into semantic chunks.

        A stable ``doc_id`` is computed from the file name and the concatenated
        text of all pages, so re-ingesting the same document yields the same
        identifiers (enabling deduplication).

        Args:
            pages: Page-level document dicts from :class:`DocumentLoader`.

        Returns:
            A list of chunk dicts ``{chunk_id, text, doc_id, chunk_index,
            page_num, file_name, block_type, section}``.
        """
        if not pages:
            return []

        file_name = pages[0].get("file_name", "unknown")
        full_text = "\n".join(page["text"] for page in pages)
        doc_id = hashlib.sha256(
            (file_name + full_text).encode("utf-8")
        ).hexdigest()[:16]

        chunks: list[dict[str, Any]] = []
        index = 0
        for page in pages:
            page_num = page.get("page_num", 1)
            blocks = self._parse_blocks(page.get("text", ""))
            index = self._pack_blocks(
                blocks, page_num, doc_id, file_name, chunks, index
            )

        n_tables = sum(1 for c in chunks if c["block_type"] == "table")
        n_images = sum(1 for c in chunks if c["block_type"] == "image")
        logger.info(
            f"Chunked doc {doc_id} into {len(chunks)} semantic chunks "
            f"(size={self.chunk_size} tokens, overlap={self.overlap}; "
            f"{n_tables} tables, {n_images} images kept separate)"
        )
        return chunks

    # ------------------------------------------------------------------
    # Block parsing
    # ------------------------------------------------------------------
    def _parse_blocks(self, text: str) -> list[tuple[str, str]]:
        """Parse raw page text into ordered ``(block_type, block_text)`` blocks.

        Block types: ``heading``, ``text``, ``table``, ``image``, ``code``.

        Args:
            text: The raw text of a page.

        Returns:
            An ordered list of blocks.
        """
        lines = text.split("\n")
        blocks: list[tuple[str, str]] = []
        para: list[str] = []

        def flush_para() -> None:
            if para:
                joined = "\n".join(para).strip()
                if joined:
                    blocks.append(("text", joined))
                para.clear()

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped = line.strip()

            # Fenced code block: keep everything until the closing fence.
            if stripped.startswith("```"):
                flush_para()
                code = [line]
                i += 1
                while i < n and not lines[i].strip().startswith("```"):
                    code.append(lines[i])
                    i += 1
                if i < n:
                    code.append(lines[i])
                    i += 1
                blocks.append(("code", "\n".join(code).strip()))
                continue

            # Blank line ends the current paragraph.
            if not stripped:
                flush_para()
                i += 1
                continue

            # Markdown image on its own line.
            if _MD_IMAGE_RE.match(stripped):
                flush_para()
                blocks.append(("image", stripped))
                i += 1
                continue

            # Heading.
            if self._is_heading(stripped):
                flush_para()
                blocks.append(("heading", stripped))
                i += 1
                continue

            # Table: a run of consecutive pipe-delimited rows.
            if self._is_table_row(line):
                run = [line]
                j = i + 1
                while j < n and self._is_table_row(lines[j]):
                    run.append(lines[j])
                    j += 1
                if len(run) >= 2:  # a single stray '|' line is not a table
                    flush_para()
                    blocks.append(("table", "\n".join(run).strip()))
                    i = j
                    continue

            # Ordinary prose line.
            para.append(line)
            i += 1

        flush_para()
        return blocks

    @staticmethod
    def _is_heading(stripped: str) -> bool:
        """Return True if a line looks like a section heading."""
        if _MD_HEADING_RE.match(stripped):
            return True
        if len(stripped) > 80:
            return False
        if _NUM_HEADING_RE.match(stripped):
            return True
        return stripped.lower().rstrip(":") in _SECTION_WORDS

    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Return True if a line looks like a (markdown) table row."""
        return line.count("|") >= 2

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------
    def _pack_blocks(
        self,
        blocks: list[tuple[str, str]],
        page_num: int,
        doc_id: str,
        file_name: str,
        chunks: list[dict[str, Any]],
        index: int,
    ) -> int:
        """Pack parsed blocks into chunks and append them to ``chunks``.

        Args:
            blocks: Parsed blocks for one page.
            page_num: The source page number.
            doc_id: The document identifier.
            file_name: The original file name.
            chunks: The output list to append to.
            index: The next chunk index to use.

        Returns:
            The updated next chunk index.
        """
        section = ""
        buf: list[str] = []  # accumulated sentences for the current prose chunk
        buf_words = 0
        buf_section = ""

        def emit(text: str, block_type: str, sec: str) -> None:
            nonlocal index
            content = text.strip()
            if not content:
                return
            # Prepend the section heading to prose chunks for retrieval context.
            stored = f"{sec}\n\n{content}" if sec and block_type == "text" else content
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{index}",
                    "text": stored,
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "page_num": page_num,
                    "file_name": file_name,
                    "block_type": block_type,
                    "section": sec,
                }
            )
            index += 1

        def flush_prose() -> None:
            nonlocal buf, buf_words
            if not buf:
                return
            emit(" ".join(buf), "text", buf_section)
            # Carry whole trailing sentences as overlap (never split a word).
            tail: list[str] = []
            tail_words = 0
            for sentence in reversed(buf):
                w = len(sentence.split())
                if tail_words + w > self._overlap_words:
                    break
                tail.insert(0, sentence)
                tail_words += w
            buf = tail
            buf_words = tail_words

        for block_type, block_text in blocks:
            if block_type == "heading":
                flush_prose()
                buf, buf_words = [], 0  # sections don't overlap into each other
                section = block_text.lstrip("#").strip()
                continue

            if block_type in ("table", "image", "code"):
                # Atomic block: flush prose, then emit it on its own.
                flush_prose()
                buf, buf_words = [], 0
                emit(block_text, block_type, section)
                continue

            # Prose block: split into sentences and pack.
            for sentence in self._split_sentences(block_text):
                words = len(sentence.split())
                if words > self._max_words:
                    # A single over-long sentence: flush, then hard-split it.
                    flush_prose()
                    buf, buf_words = [], 0
                    self._emit_long_sentence(
                        sentence, section, emit
                    )
                    continue
                if buf and buf_words + words > self._max_words:
                    flush_prose()
                if not buf:
                    buf_section = section
                buf.append(sentence)
                buf_words += words

        flush_prose()
        return index

    def _emit_long_sentence(self, sentence: str, section: str, emit: Any) -> None:
        """Hard-split a single sentence longer than the budget, by words."""
        words = sentence.split()
        for start in range(0, len(words), self._max_words):
            emit(" ".join(words[start : start + self._max_words]), "text", section)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split a prose block into sentences.

        Newlines inside a paragraph are treated as soft breaks (collapsed to
        spaces) before splitting on sentence-ending punctuation.

        Args:
            text: A prose block.

        Returns:
            A list of sentence strings (never empty strings).
        """
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        parts = _SENTENCE_RE.split(normalized)
        return [part.strip() for part in parts if part.strip()]
