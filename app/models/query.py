"""Pydantic models for the query API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class QueryMode(str, Enum):
    """Supported retrieval modes."""

    HYBRID = "hybrid"
    VECTOR = "vector"
    GRAPH = "graph"


class QueryRequest(BaseModel):
    """Request body for a question-answering query."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=4000,
        description="The natural-language question to answer.",
    )
    mode: QueryMode = Field(
        default=QueryMode.HYBRID, description="Retrieval strategy to use."
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of chunks to retrieve."
    )


class Source(BaseModel):
    """A single source chunk cited in an answer."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(description="Identifier of the source chunk.")
    text: str = Field(description="Raw text of the source chunk.")
    doc_id: str = Field(description="Parent document identifier.")
    score: float | None = Field(
        default=None, description="Vector similarity score, if available."
    )
    source: str = Field(
        description="Retrieval origin: 'vector', 'graph', or 'both'."
    )


class QueryResponse(BaseModel):
    """Response returned for a query."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="The generated answer text.")
    sources: list[Source] = Field(
        default_factory=list, description="Chunks used to ground the answer."
    )
    graph_context: list[dict] = Field(
        default_factory=list,
        description="Relationship paths discovered during graph retrieval.",
    )
    mode_used: str = Field(description="The retrieval mode actually used.")
    entities_found: list[str] = Field(
        default_factory=list,
        description="Entities from the question found in the graph.",
    )
