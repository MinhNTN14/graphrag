"""FastAPI application entry point.

Wires together the Neo4j client, ingestion pipeline, retrievers and answer
generator into a single application, exposing ingest/query/graph routes plus a
health check. Singletons live on ``app.state``.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.routes import graph as graph_routes
from app.api.routes import ingest as ingest_routes
from app.api.routes import query as query_routes
from app.config import get_settings
from app.core.generation.answer_generator import AnswerGenerator
from app.core.graph.graph_builder import GraphBuilder
from app.core.graph.neo4j_client import Neo4jClient, Neo4jError
from app.core.ingestion.chunker import TextChunker
from app.core.ingestion.document_loader import DocumentLoader
from app.core.ingestion.embedder import GeminiEmbedder
from app.core.ingestion.entity_extractor import EntityExtractor
from app.core.retrieval.graph_retriever import GraphRetriever
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.vector_retriever import VectorRetriever


def _configure_logging() -> None:
    """Configure loguru according to the ``LOG_LEVEL`` setting."""
    settings = get_settings()
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and tear down shared singletons.

    Args:
        app: The FastAPI application instance.
    """
    _configure_logging()
    settings = get_settings()
    logger.info("Starting GraphRAG API")

    neo4j_client = Neo4jClient(settings)
    embedder = GeminiEmbedder(settings)
    extractor = EntityExtractor(settings)

    document_loader = DocumentLoader()
    chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    graph_builder = GraphBuilder(neo4j_client, embedder, extractor)

    vector_retriever = VectorRetriever(neo4j_client, embedder)
    graph_retriever = GraphRetriever(neo4j_client, extractor)
    hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever)
    answer_generator = AnswerGenerator(settings)

    app.state.neo4j_client = neo4j_client
    app.state.embedder = embedder
    app.state.extractor = extractor
    app.state.document_loader = document_loader
    app.state.chunker = chunker
    app.state.graph_builder = graph_builder
    app.state.vector_retriever = vector_retriever
    app.state.graph_retriever = graph_retriever
    app.state.hybrid_retriever = hybrid_retriever
    app.state.answer_generator = answer_generator

    try:
        yield
    finally:
        logger.info("Shutting down GraphRAG API")
        neo4j_client.close()


app = FastAPI(title="GraphRAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_routes.router, prefix="/api/v1")
app.include_router(query_routes.router, prefix="/api/v1")
app.include_router(graph_routes.router, prefix="/api/v1")


@app.exception_handler(Neo4jError)
async def _neo4j_error_handler(request: Request, exc: Neo4jError) -> JSONResponse:
    """Return HTTP 503 when a Neo4j operation fails.

    Args:
        request: The incoming request.
        exc: The raised :class:`Neo4jError`.

    Returns:
        A JSON error response.
    """
    logger.error(f"Neo4jError on {request.url.path}: {exc}")
    return JSONResponse(status_code=503, content={"detail": f"Database error: {exc}"})


@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Return HTTP 400 for uncaught ``ValueError`` instances.

    Args:
        request: The incoming request.
        exc: The raised :class:`ValueError`.

    Returns:
        A JSON error response.
    """
    logger.error(f"ValueError on {request.url.path}: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, object]:
    """Return service health, including Neo4j connectivity.

    Returns:
        A dict ``{status, neo4j}``.
    """
    neo4j_ok = False
    client = getattr(app.state, "neo4j_client", None)
    if client is not None:
        neo4j_ok = client.health_check()
    return {"status": "ok", "neo4j": neo4j_ok}
