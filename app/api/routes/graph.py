"""Knowledge-graph exploration API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.core.graph.neo4j_client import Neo4jClient
from app.core.ingestion.embedder import GeminiEmbedder
from app.models.entity import (
    EntityResponse,
    RelationResponse,
    SubgraphResponse,
)

router = APIRouter(prefix="/graph", tags=["graph"])


class EmbedQueryRequest(BaseModel):
    """Request body for embedding a query and finding nearest chunks."""

    question: str
    top_k: int = 5


@router.get("/entities", response_model=list[EntityResponse])
async def list_entities(
    request: Request,
    type: str | None = Query(default=None, description="Filter by entity type."),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[EntityResponse]:
    """List entities, optionally filtered by type.

    Args:
        request: The FastAPI request.
        type: Optional entity type filter.
        limit: Maximum number of entities to return.

    Returns:
        A list of entities.
    """
    client: Neo4jClient = request.app.state.neo4j_client
    entities = client.get_all_entities(type_filter=type, limit=limit)
    return [EntityResponse(**e) for e in entities]


@router.get("/entity/{name}/neighbors", response_model=SubgraphResponse)
async def entity_neighbors(
    request: Request,
    name: str,
    depth: int = Query(default=2, ge=1, le=4),
) -> SubgraphResponse:
    """Return the subgraph surrounding a named entity.

    Args:
        request: The FastAPI request.
        name: The centre entity name.
        depth: Number of hops to include.

    Returns:
        The subgraph around the entity.

    Raises:
        HTTPException: 404 if the entity does not exist.
    """
    client: Neo4jClient = request.app.state.neo4j_client
    subgraph = client.get_entity_neighbors(name, depth=depth)
    if subgraph["entity"] is None:
        raise HTTPException(status_code=404, detail=f"Entity not found: {name}")
    return SubgraphResponse(
        entity=EntityResponse(**subgraph["entity"]),
        neighbors=[EntityResponse(**n) for n in subgraph["neighbors"]],
        relationships=[RelationResponse(**r) for r in subgraph["relationships"]],
    )


@router.get("/relations", response_model=list[RelationResponse])
async def relation_paths(
    request: Request,
    source: str = Query(..., description="Source entity name."),
    target: str = Query(..., description="Target entity name."),
) -> list[RelationResponse]:
    """Return the shortest relationship path between two entities.

    Args:
        request: The FastAPI request.
        source: Source entity name.
        target: Target entity name.

    Returns:
        A list of relationships forming the path (empty if none exists).
    """
    client: Neo4jClient = request.app.state.neo4j_client
    relations = client.find_relation_paths(source, target)
    return [RelationResponse(**r) for r in relations]


# ----------------------------------------------------------------------
# Pipeline-visualisation endpoints
# ----------------------------------------------------------------------
@router.get("/documents")
async def list_documents(request: Request) -> list[dict[str, Any]]:
    """List ingested documents with chunk and entity counts.

    Args:
        request: The FastAPI request.

    Returns:
        A list of document summary dicts.
    """
    client: Neo4jClient = request.app.state.neo4j_client
    return client.get_documents()


@router.get("/document/{doc_id}")
async def document_graph(request: Request, doc_id: str) -> dict[str, Any]:
    """Return a document's chunks (with embeddings) and entity relationships.

    Args:
        request: The FastAPI request.
        doc_id: The document identifier.

    Returns:
        A dict with ``chunks`` and ``relationships`` for visualisation.

    Raises:
        HTTPException: 404 if the document has no chunks.
    """
    client: Neo4jClient = request.app.state.neo4j_client
    graph = client.get_document_graph(doc_id)
    if not graph["chunks"]:
        raise HTTPException(status_code=404, detail=f"No chunks for doc: {doc_id}")
    return graph


@router.post("/embed-query")
async def embed_query(request: Request, body: EmbedQueryRequest) -> dict[str, Any]:
    """Embed a question and return its vector plus nearest chunks.

    Used by the visualiser to place a user query into the same embedding
    space as the document chunks and highlight what it retrieves.

    Args:
        request: The FastAPI request.
        body: The query and desired number of neighbours.

    Returns:
        A dict ``{embedding, retrieved: [{chunk_id, score, doc_id}]}``.
    """
    client: Neo4jClient = request.app.state.neo4j_client
    embedder: GeminiEmbedder = request.app.state.embedder
    embedding = embedder.embed(body.question)
    retrieved = client.vector_search(embedding, body.top_k)
    return {
        "embedding": embedding,
        "retrieved": [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "score": r["score"],
            }
            for r in retrieved
        ],
    }
