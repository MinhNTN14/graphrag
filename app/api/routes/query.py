"""Query API routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from app.core.generation.answer_generator import AnswerGenerator
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.models.query import QueryRequest, QueryResponse, Source

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def run_query(request: Request, body: QueryRequest) -> QueryResponse:
    """Answer a question using the selected retrieval mode.

    Args:
        request: The FastAPI request, used to access shared app state.
        body: The query request body.

    Returns:
        The generated answer with sources and graph context.
    """
    retriever: HybridRetriever = request.app.state.hybrid_retriever
    generator: AnswerGenerator = request.app.state.answer_generator

    # Retrieval and generation perform blocking network I/O; run off-loop.
    retrieval_result = await asyncio.to_thread(
        retriever.retrieve, body.question, body.mode.value, body.top_k
    )
    result = await asyncio.to_thread(
        generator.generate, body.question, retrieval_result
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        graph_context=result["graph_context"],
        mode_used=result["mode_used"],
        entities_found=result["entities_found"],
    )
