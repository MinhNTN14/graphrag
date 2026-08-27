"""Document loading utilities.

Supports loading PDF, TXT and Markdown documents either from a filesystem path
or from raw bytes (for API uploads). Each loader returns a list of page-level
document dicts of the form ``{text, page_num, file_name}``.

PDFs are extracted structurally: ``pdfplumber`` detects tables and pulls prose
from *outside* the table regions, and ``PyMuPDF`` counts embedded images. Tables
are emitted as markdown pipe tables and images as markdown image markers, so the
downstream :class:`~app.core.ingestion.chunker.TextChunker` keeps each as its own
atomic chunk. If the rich path fails for any reason we fall back to plain
``pypdf`` text extraction.
"""

from __future__ import annotations

import io
import os
from typing import Any

from loguru import logger
from pypdf import PdfReader


class UnsupportedFileTypeError(Exception):
    """Raised when an unsupported file extension is provided."""


_TEXT_EXTENSIONS = {".txt", ".md"}
_PDF_EXTENSIONS = {".pdf"}


class DocumentLoader:
    """Load documents from disk or bytes into page-level records."""

    def load(self, file_path: str) -> list[dict[str, Any]]:
        """Load a document from a filesystem path.

        Args:
            file_path: Path to a ``.pdf``, ``.txt`` or ``.md`` file.

        Returns:
            A list of page dicts ``{text, page_num, file_name}``.

        Raises:
            UnsupportedFileTypeError: If the extension is not supported.
            FileNotFoundError: If the path does not exist.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        logger.info(f"Loading document: {file_name} (ext={ext})")

        if ext in _PDF_EXTENSIONS:
            with open(file_path, "rb") as handle:
                return self._load_pdf(handle.read(), file_name)
        if ext in _TEXT_EXTENSIONS:
            with open(file_path, encoding="utf-8", errors="replace") as handle:
                return self._load_text(handle.read(), file_name)
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")

    def load_from_bytes(self, content: bytes, filename: str) -> list[dict[str, Any]]:
        """Load a document from raw bytes (e.g. an uploaded file).

        Args:
            content: Raw file bytes.
            filename: Original file name used to detect the type.

        Returns:
            A list of page dicts ``{text, page_num, file_name}``.

        Raises:
            UnsupportedFileTypeError: If the extension is not supported.
        """
        ext = os.path.splitext(filename)[1].lower()
        file_name = os.path.basename(filename)
        logger.info(f"Loading document from bytes: {file_name} (ext={ext})")

        if ext in _PDF_EXTENSIONS:
            return self._load_pdf(content, file_name)
        if ext in _TEXT_EXTENSIONS:
            return self._load_text(content.decode("utf-8", errors="replace"), file_name)
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")

    # ------------------------------------------------------------------
    # PDF loading
    # ------------------------------------------------------------------
    def _load_pdf(self, content: bytes, file_name: str) -> list[dict[str, Any]]:
        """Extract a PDF structurally, falling back to plain text on failure.

        Args:
            content: Raw PDF bytes.
            file_name: Original file name.

        Returns:
            A list of per-page dicts. Empty pages are skipped.
        """
        try:
            pages = self._load_pdf_rich(content, file_name)
            if pages:
                return pages
            logger.warning(f"Rich PDF extraction yielded no text for {file_name}")
        except Exception as exc:  # noqa: BLE001 - any failure -> safe fallback
            logger.warning(
                f"Rich PDF extraction failed for {file_name} ({exc}); "
                "falling back to pypdf"
            )
        return self._load_pdf_basic(content, file_name)

    def _load_pdf_rich(self, content: bytes, file_name: str) -> list[dict[str, Any]]:
        """Structure-aware PDF extraction (prose + tables + image markers).

        Args:
            content: Raw PDF bytes.
            file_name: Original file name.

        Returns:
            A list of per-page dicts whose text embeds markdown tables and image
            markers separated by blank lines. Empty pages are skipped.
        """
        import fitz  # PyMuPDF, imported lazily so the module loads without it
        import pdfplumber

        # Count embedded images per page with PyMuPDF (pdfplumber sees them too,
        # but PyMuPDF's image table is more reliable for XObject figures).
        with fitz.open(stream=content, filetype="pdf") as doc:
            image_counts = [len(doc[i].get_images(full=True)) for i in range(len(doc))]

        pages: list[dict[str, Any]] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for index, page in enumerate(pdf.pages):
                image_count = image_counts[index] if index < len(image_counts) else 0
                text = self._compose_pdf_page(page, image_count, index + 1, file_name)
                if text:
                    pages.append(
                        {"text": text, "page_num": index + 1, "file_name": file_name}
                    )
        logger.info(
            f"Rich-extracted {len(pages)} non-empty pages from {file_name} "
            f"({sum(image_counts)} embedded images detected)"
        )
        return pages

    def _compose_pdf_page(
        self, page: Any, image_count: int, page_num: int, file_name: str
    ) -> str:
        """Compose one PDF page into prose + separated tables + image markers.

        Prose is taken from words lying *outside* every detected table's bounding
        box (so table cells are not duplicated in the surrounding text). Each
        table is rendered as a markdown pipe table and each embedded image as a
        markdown image marker; blocks are joined by blank lines so the chunker
        parses them as separate atomic blocks.

        Args:
            page: A ``pdfplumber`` page.
            image_count: Number of embedded images on the page.
            page_num: 1-based page number.
            file_name: Original file name (used in image marker URLs).

        Returns:
            The composed page text (may be empty).
        """
        # pdfplumber over-detects tables on multi-column layouts (it reads the
        # column gutter as a table). Keep only tables that actually look like
        # tables so we don't (a) emit garbage table blocks or (b) strip real
        # body text out of the prose by excluding a spurious table region.
        real_tables = [
            t for t in page.find_tables() if self._looks_like_real_table(t.extract())
        ]
        table_boxes = [t.bbox for t in real_tables]

        if table_boxes:
            prose = (
                page.filter(lambda obj: self._outside_boxes(obj, table_boxes))
                .extract_text()
                or ""
            )
        else:
            prose = page.extract_text() or ""

        parts: list[str] = [prose.strip()]

        for table in real_tables:
            markdown = self._table_to_markdown(table.extract())
            if markdown.strip():
                parts.append(markdown.strip())

        for i in range(image_count):
            parts.append(
                f"![figure on page {page_num} (image {i + 1})]"
                f"(pdf://{file_name}#page{page_num}-img{i + 1})"
            )

        return "\n\n".join(part for part in parts if part).strip()

    @staticmethod
    def _outside_boxes(obj: dict[str, Any], boxes: list[tuple]) -> bool:
        """Return True if a pdfplumber object's centre lies outside all boxes.

        Args:
            obj: A pdfplumber object with ``x0``/``x1``/``top``/``bottom`` keys.
            boxes: Table bounding boxes as ``(x0, top, x1, bottom)`` tuples.

        Returns:
            True if the object is not inside any table region.
        """
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return all(
            not (x0 <= cx <= x1 and top <= cy <= bottom)
            for x0, top, x1, bottom in boxes
        )

    @staticmethod
    def _looks_like_real_table(rows: list[list[str | None]]) -> bool:
        """Heuristic gate to reject spurious tables from multi-column layouts.

        A genuine table has several rows, at least two columns and a reasonable
        proportion of filled cells. Column gutters misread as tables tend to be
        mostly empty, so we require both an absolute and a relative fill level.

        Args:
            rows: The extracted table as a list of rows of (optional) cells.

        Returns:
            True if the table looks real enough to keep.
        """
        if not rows or len(rows) < 2:
            return False
        ncols = max((len(row) for row in rows), default=0)
        if ncols < 2:
            return False
        total = sum(len(row) for row in rows) or 1
        filled = sum(1 for row in rows for cell in row if cell and cell.strip())
        return filled >= 4 and filled / total >= 0.4

    @staticmethod
    def _table_to_markdown(rows: list[list[str | None]]) -> str:
        """Convert an extracted table (rows of cells) into a markdown pipe table.

        ``rows`` is a list of rows; each row is a list of cell strings, where a
        missing/empty cell may be ``None``. The result must be a valid markdown
        pipe table so the chunker recognises it as a table block:

            | Header A | Header B |
            | --- | --- |
            | a1 | b1 |

        Contract for the chunker to detect it:
          * At least 2 output lines, each containing at least two ``|`` chars.
          * The first ``rows`` entry is treated as the header, followed by a
            ``| --- | ... |`` separator line, then one line per remaining row.

        Args:
            rows: The extracted table as a list of rows of (optional) cells.

        Returns:
            A markdown pipe-table string, or ``""`` if the table is unusable
            (empty, or fewer than 2 columns / no rows).
        """
        if not rows or len(rows[0]) < 2:
            return ""

        def cell(value: str | None) -> str:
            # None -> ""; collapse internal newlines; escape pipes.
            return " ".join((value or "").split()).replace("|", "\\|")

        ncols = max(len(row) for row in rows)
        lines: list[str] = []
        for r_index, row in enumerate(rows):
            cells = [cell(row[i] if i < len(row) else "") for i in range(ncols)]
            lines.append("| " + " | ".join(cells) + " |")
            if r_index == 0:  # separator directly under the header row
                lines.append("| " + " | ".join(["---"] * ncols) + " |")
        return "\n".join(lines)

    def _load_pdf_basic(self, content: bytes, file_name: str) -> list[dict[str, Any]]:
        """Plain-text PDF extraction with pypdf, one record per page.

        Used as a fallback when structure-aware extraction is unavailable.

        Args:
            content: Raw PDF bytes.
            file_name: Original file name.

        Returns:
            A list of per-page dicts. Empty pages are skipped.
        """
        reader = PdfReader(io.BytesIO(content))
        pages: list[dict[str, Any]] = []
        for index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pages.append(
                {"text": text, "page_num": index + 1, "file_name": file_name}
            )
        logger.info(f"Extracted {len(pages)} non-empty pages from {file_name} (pypdf)")
        return pages

    # ------------------------------------------------------------------
    # Text loading
    # ------------------------------------------------------------------
    def _load_text(self, text: str, file_name: str) -> list[dict[str, Any]]:
        """Wrap a plain-text/markdown document as a single-page record.

        Args:
            text: The full document text.
            file_name: Original file name.

        Returns:
            A single-element list containing the document.
        """
        return [{"text": text.strip(), "page_num": 1, "file_name": file_name}]
