"""Ingestion API routes."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.graph.graph_builder import GraphBuilder
from app.core.ingestion.chunker import TextChunker
from app.core.ingestion.document_loader import (
    DocumentLoader,
    UnsupportedFileTypeError,
)
from app.models.document import IngestRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _build_response(summary: dict, file_label: str) -> IngestResponse:
    """Construct an :class:`IngestResponse` from a builder summary.

    Args:
        summary: The dict returned by :meth:`GraphBuilder.build_from_chunks`.
        file_label: Name of the ingested file for the message.

    Returns:
        The populated response model.
    """
    if summary["chunks_processed"] == 0 and summary["doc_id"]:
        message = f"Document '{file_label}' already ingested; skipped."
    elif not summary["doc_id"]:
        message = f"No extractable content found in '{file_label}'."
    else:
        message = f"Successfully ingested '{file_label}'."
    return IngestResponse(
        doc_id=summary["doc_id"],
        chunks_processed=summary["chunks_processed"],
        entities_created=summary["entities_created"],
        relationships_created=summary["relationships_created"],
        message=message,
    )


@router.post("", response_model=IngestResponse)
async def ingest_upload(
    request: Request, file: UploadFile = File(...)
) -> IngestResponse:
    """Ingest an uploaded document (PDF, TXT or MD).

    Args:
        request: The FastAPI request, used to access shared app state.
        file: The uploaded file.

    Returns:
        Ingestion statistics.
    """
    loader: DocumentLoader = request.app.state.document_loader
    chunker: TextChunker = request.app.state.chunker
    builder: GraphBuilder = request.app.state.graph_builder

    content = await file.read()
    try:
        pages = loader.load_from_bytes(content, file.filename or "upload")
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = chunker.chunk(pages)
    summary = await builder.build_from_chunks(chunks)
    return _build_response(summary, file.filename or "upload")


@router.post("/path", response_model=IngestResponse)
async def ingest_path(request: Request, body: IngestRequest) -> IngestResponse:
    """Ingest a document from a filesystem path.

    Args:
        request: The FastAPI request, used to access shared app state.
        body: The request body containing ``file_path``.

    Returns:
        Ingestion statistics.
    """
    if not body.file_path:
        raise HTTPException(status_code=400, detail="file_path is required")

    loader: DocumentLoader = request.app.state.document_loader
    chunker: TextChunker = request.app.state.chunker
    builder: GraphBuilder = request.app.state.graph_builder

    try:
        pages = loader.load(body.file_path)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    chunks = chunker.chunk(pages)
    summary = await builder.build_from_chunks(chunks)
    return _build_response(summary, body.file_path)
