"""Graph-based retrieval driven by entities extracted from the question."""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.core.graph.neo4j_client import Neo4jClient
from app.core.ingestion.entity_extractor import EntityExtractor


class GraphRetriever:
    """Retrieve context by traversing the knowledge graph."""

    def __init__(self, neo4j_client: Neo4jClient, extractor: EntityExtractor) -> None:
        """Store the Neo4j client and entity extractor.

        Args:
            neo4j_client: Client exposing ``graph_search``.
            extractor: Extractor used to pull entities from the question.
        """
        self._client = neo4j_client
        self._extractor = extractor

    def retrieve(self, question: str, top_k: int) -> dict[str, Any]:
        """Retrieve graph context relevant to ``question``.

        Extracts entities from the question, traverses the graph up to two hops
        and collects the chunks that mention reached entities.

        Args:
            question: The natural-language question.
            top_k: Maximum number of chunks to keep.

        Returns:
            A dict with ``chunks``, ``paths`` and ``entities_found``.
        """
        logger.debug(f"Graph retrieval for question: {question!r} (top_k={top_k})")
        extraction = self._extractor.extract(question)
        entity_names = [
            (e.get("name") or "").strip()
            for e in extraction.get("entities", [])
            if (e.get("name") or "").strip()
        ]
        if not entity_names:
            logger.info("No entities extracted from question; graph retrieval empty")
            return {"chunks": [], "paths": [], "entities_found": []}

        result = self._client.graph_search(entity_names, depth=2)
        for chunk in result["chunks"]:
            chunk["source"] = "graph"
        result["chunks"] = result["chunks"][:top_k]
        return result
