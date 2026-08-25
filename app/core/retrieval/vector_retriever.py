"""Vector-based retrieval over chunk embeddings."""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.core.graph.neo4j_client import Neo4jClient
from app.core.ingestion.embedder import GeminiEmbedder


class VectorRetriever:
    """Retrieve chunks by embedding-similarity to the question."""

    def __init__(self, neo4j_client: Neo4jClient, embedder: GeminiEmbedder) -> None:
        """Store the Neo4j client and embedder.

        Args:
            neo4j_client: Client exposing ``vector_search``.
            embedder: Embedding provider for the question.
        """
        self._client = neo4j_client
        self._embedder = embedder

    def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        """Return the ``top_k`` chunks most similar to ``question``.

        Args:
            question: The natural-language question.
            top_k: Number of chunks to return.

        Returns:
            A list of chunk dicts ``{chunk_id, text, doc_id, score}``.
        """
        logger.debug(f"Vector retrieval for question: {question!r} (top_k={top_k})")
        embedding = self._embedder.embed(question)
        results = self._client.vector_search(embedding, top_k)
        for result in results:
            result["source"] = "vector"
        return results
