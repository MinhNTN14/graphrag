"""Pydantic models for the document ingestion API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IngestRequest(BaseModel):
    """Request body for ingesting a document by filesystem path."""

    model_config = ConfigDict(extra="forbid")

    file_path: str | None = Field(
        default=None, description="Path to a local PDF/TXT/MD file to ingest."
    )


class IngestResponse(BaseModel):
    """Response returned after a document has been ingested."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="Stable identifier of the ingested document.")
    chunks_processed: int = Field(description="Number of chunks written.")
    entities_created: int = Field(description="Number of entity upserts performed.")
    relationships_created: int = Field(
        description="Number of relationship upserts performed."
    )
    message: str = Field(description="Human-readable status message.")
