"""Hybrid retrieval combining vector similarity and graph traversal."""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.core.retrieval.graph_retriever import GraphRetriever
from app.core.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    """Dispatch to vector, graph, or a merged hybrid retrieval strategy."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
    ) -> None:
        """Store the underlying retrievers.

        Args:
            vector_retriever: The vector-similarity retriever.
            graph_retriever: The graph-traversal retriever.
        """
        self._vector = vector_retriever
        self._graph = graph_retriever

    def retrieve(self, question: str, mode: str, top_k: int) -> dict[str, Any]:
        """Retrieve context according to the selected ``mode``.

        Args:
            question: The natural-language question.
            mode: One of ``"vector"``, ``"graph"`` or ``"hybrid"``.
            top_k: Number of chunks to target.

        Returns:
            A dict with ``chunks``, ``graph_context``, ``mode_used`` and
            ``entities_found``.
        """
        logger.info(f"Hybrid retrieve mode={mode} top_k={top_k}")
        if mode == "vector":
            chunks = self._vector.retrieve(question, top_k)
            return {
                "chunks": chunks,
                "graph_context": [],
                "mode_used": "vector",
                "entities_found": [],
            }
        if mode == "graph":
            graph = self._graph.retrieve(question, top_k)
            return {
                "chunks": graph["chunks"],
                "graph_context": graph["paths"],
                "mode_used": "graph",
                "entities_found": graph["entities_found"],
            }

        # --- hybrid --------------------------------------------------
        vector_chunks = self._vector.retrieve(question, top_k)
        graph = self._graph.retrieve(question, top_k)
        graph_chunks = graph["chunks"]

        merged = self._merge_chunks(vector_chunks, graph_chunks)
        return {
            "chunks": merged,
            "graph_context": graph["paths"],
            "mode_used": "hybrid",
            "entities_found": graph["entities_found"],
        }

    def _merge_chunks(
        self,
        vector_chunks: list[dict[str, Any]],
        graph_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Deduplicate and merge chunks from both retrieval strategies.

        Chunks appearing in both sources are flagged ``source="both"``. The
        final list is sorted by descending vector score (chunks without a score
        sort last).

        Args:
            vector_chunks: Chunks from the vector retriever.
            graph_chunks: Chunks from the graph retriever.

        Returns:
            The merged, deduplicated and ranked chunk list.
        """
        by_id: dict[str, dict[str, Any]] = {}
        for chunk in vector_chunks:
            by_id[chunk["chunk_id"]] = dict(chunk)

        for chunk in graph_chunks:
            cid = chunk["chunk_id"]
            if cid in by_id:
                by_id[cid]["source"] = "both"
            else:
                new_chunk = dict(chunk)
                new_chunk["source"] = "graph"
                by_id[cid] = new_chunk

        def sort_key(item: dict[str, Any]) -> float:
            score = item.get("score")
            return score if isinstance(score, int | float) else -1.0

        merged = sorted(by_id.values(), key=sort_key, reverse=True)
        logger.debug(
            f"Merged {len(vector_chunks)} vector + {len(graph_chunks)} graph "
            f"chunks into {len(merged)} unique chunks"
        )
        return merged
