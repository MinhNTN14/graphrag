"""Graph construction from document chunks.

Orchestrates embedding, chunk upsert, entity/relationship extraction and
graph linking. Deduplicates whole documents via a ``:Document`` marker node.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from app.core.graph.neo4j_client import Neo4jClient
from app.core.ingestion.embedder import GeminiEmbedder
from app.core.ingestion.entity_extractor import EntityExtractor


class GraphBuilder:
    """Build the knowledge graph from a list of text chunks."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        embedder: GeminiEmbedder,
        extractor: EntityExtractor,
    ) -> None:
        """Store collaborators used during graph construction.

        Args:
            neo4j_client: Client used to persist nodes and relationships.
            embedder: Embedding provider for chunk vectors.
            extractor: Entity/relationship extractor.
        """
        self._client = neo4j_client
        self._embedder = embedder
        self._extractor = extractor

    async def build_from_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Process chunks into the knowledge graph.

        For each chunk this embeds the text, upserts the chunk, extracts
        entities/relationships and links the chunk to each entity. Documents
        already fully ingested are skipped.

        Args:
            chunks: Chunk dicts from :class:`TextChunker`.

        Returns:
            A summary dict with ``doc_id``, ``chunks_processed``,
            ``entities_created`` and ``relationships_created``.
        """
        if not chunks:
            return {
                "doc_id": "",
                "chunks_processed": 0,
                "entities_created": 0,
                "relationships_created": 0,
            }

        doc_id = chunks[0]["doc_id"]
        if self._client.document_exists(doc_id):
            logger.info(f"Document {doc_id} already ingested; skipping")
            return {
                "doc_id": doc_id,
                "chunks_processed": 0,
                "entities_created": 0,
                "relationships_created": 0,
            }

        entities_created = 0
        relationships_created = 0
        chunks_processed = 0

        for chunk in chunks:
            # Offload blocking network calls to worker threads so the async
            # endpoint remains responsive.
            embedding = await asyncio.to_thread(self._embedder.embed, chunk["text"])
            await asyncio.to_thread(
                self._client.upsert_chunk,
                chunk["chunk_id"],
                chunk["text"],
                embedding,
                chunk["doc_id"],
                chunk["chunk_index"],
                chunk.get("page_num", 0),
                chunk.get("file_name", ""),
                chunk.get("block_type", "text"),
                chunk.get("section", ""),
            )

            extraction = await asyncio.to_thread(self._extractor.extract, chunk["text"])
            entities = extraction.get("entities", [])
            relationships = extraction.get("relationships", [])

            for entity in entities:
                name = (entity.get("name") or "").strip()
                if not name:
                    continue
                await asyncio.to_thread(
                    self._client.upsert_entity,
                    name,
                    entity.get("type", "CONCEPT"),
                    entity.get("description", ""),
                )
                await asyncio.to_thread(
                    self._client.link_chunk_to_entity, chunk["chunk_id"], name
                )
                entities_created += 1

            for rel in relationships:
                source = (rel.get("source") or "").strip()
                target = (rel.get("target") or "").strip()
                relation = (rel.get("relation") or "").strip()
                if not (source and target and relation):
                    continue
                await asyncio.to_thread(
                    self._client.upsert_relationship,
                    source,
                    relation,
                    target,
                    rel.get("description", ""),
                )
                relationships_created += 1

            chunks_processed += 1

        await asyncio.to_thread(self._client.mark_document_ingested, doc_id)
        logger.info(
            f"Built graph for doc {doc_id}: {chunks_processed} chunks, "
            f"{entities_created} entities, {relationships_created} relationships"
        )
        return {
            "doc_id": doc_id,
            "chunks_processed": chunks_processed,
            "entities_created": entities_created,
            "relationships_created": relationships_created,
        }
