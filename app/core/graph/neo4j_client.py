"""Neo4j client encapsulating all graph database operations.

The client owns the driver lifecycle, initialises the schema (constraints,
vector index and full-text index) and exposes typed helper methods for
upserting nodes/relationships and running vector + graph retrieval queries.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError as DriverNeo4jError

from app.config import Settings, get_settings


class Neo4jError(Exception):
    """Raised when a Neo4j operation fails."""


# Schema statements executed once at start-up. Each is idempotent.
_SCHEMA_STATEMENTS: list[str] = [
    "CREATE CONSTRAINT entity_name IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
    "FOR (c:Chunk) ON (c.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 768, "
    "`vector.similarity_function`: 'cosine'}}",
    "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name, e.description]",
]


class Neo4jClient:
    """Thin wrapper around the official Neo4j Python driver."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create the driver and initialise the graph schema.

        Args:
            settings: Optional settings override. Defaults to the cached
                application settings.
        """
        self._settings = settings or get_settings()
        try:
            self._driver: Driver = GraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=self._settings.neo4j_auth,
            )
        except DriverNeo4jError as exc:  # pragma: no cover - driver init
            logger.error(f"Failed to create Neo4j driver: {exc}")
            raise Neo4jError(str(exc)) from exc
        self._database = self._settings.neo4j_database
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a Cypher query and return a list of record dicts.

        Args:
            query: The Cypher statement.
            params: Query parameters.

        Returns:
            A list of records as plain dictionaries.

        Raises:
            Neo4jError: If the underlying driver raises.
        """
        params = params or {}
        logger.debug(f"Cypher: {query} | params: {params}")

        def work(tx: Any) -> list[dict[str, Any]]:
            result = tx.run(query, params)
            return [record.data() for record in result]

        try:
            with self._driver.session(database=self._database) as session:
                # Managed transaction: the driver transparently retries on
                # transient failures (SessionExpired, ServiceUnavailable, a
                # dropped Aura connection) within max_transaction_retry_time.
                return session.execute_write(work)
        except DriverNeo4jError as exc:
            logger.error(f"Neo4j query failed: {exc} | query: {query}")
            raise Neo4jError(str(exc)) from exc

    def _init_schema(self) -> None:
        """Create constraints and indexes required by the application."""
        for statement in _SCHEMA_STATEMENTS:
            try:
                self._run(statement)
            except Neo4jError as exc:
                # A failing index/constraint creation should not be fatal on
                # environments where it already exists in an incompatible form,
                # but we surface it loudly.
                logger.warning(f"Schema statement failed (continuing): {exc}")
        logger.info("Neo4j schema initialised")

    # ------------------------------------------------------------------
    # Upsert operations
    # ------------------------------------------------------------------
    def upsert_entity(self, name: str, type: str, description: str) -> None:
        """Create or update an :Entity node keyed by ``name``.

        Args:
            name: Unique entity name.
            type: Entity category (PERSON, ORGANIZATION, ...).
            description: Short description of the entity.
        """
        query = (
            "MERGE (e:Entity {name: $name}) "
            "SET e.type = $type, e.description = $description"
        )
        self._run(query, {"name": name, "type": type, "description": description})

    def upsert_relationship(
        self, source: str, relation: str, target: str, description: str
    ) -> None:
        """Create or update a RELATES_TO relationship between two entities.

        Args:
            source: Source entity name.
            relation: Relationship label stored on the edge (e.g. ``USES``).
            target: Target entity name.
            description: Short description of the relationship.
        """
        query = (
            "MERGE (a:Entity {name: $source}) "
            "MERGE (b:Entity {name: $target}) "
            "MERGE (a)-[r:RELATES_TO {relation: $relation}]->(b) "
            "SET r.description = $description"
        )
        self._run(
            query,
            {
                "source": source,
                "relation": relation,
                "target": target,
                "description": description,
            },
        )

    def upsert_chunk(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        doc_id: str,
        chunk_index: int,
        page_num: int = 0,
        file_name: str = "",
        block_type: str = "text",
        section: str = "",
    ) -> None:
        """Create or update a :Chunk node and link it to its :Document.

        Args:
            chunk_id: Unique chunk identifier.
            text: Raw chunk text.
            embedding: 768-dim embedding vector.
            doc_id: Parent document identifier.
            chunk_index: Position of the chunk within the document.
            page_num: Source page number (for visualisation labels).
            file_name: Original file name (stored on the Document node too).
            block_type: Structural type of the chunk (``text``/``table``/
                ``image``/``code``).
            section: Section heading the chunk falls under, if any.
        """
        query = (
            "MERGE (c:Chunk {id: $chunk_id}) "
            "SET c.text = $text, c.embedding = $embedding, "
            "c.doc_id = $doc_id, c.chunk_index = $chunk_index, "
            "c.page_num = $page_num, c.file_name = $file_name, "
            "c.block_type = $block_type, c.section = $section "
            "WITH c "
            "MERGE (d:Document {id: $doc_id}) "
            "SET d.file_name = $file_name "
            "MERGE (d)-[:HAS_CHUNK]->(c)"
        )
        self._run(
            query,
            {
                "chunk_id": chunk_id,
                "text": text,
                "embedding": embedding,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "page_num": page_num,
                "file_name": file_name,
                "block_type": block_type,
                "section": section,
            },
        )

    def link_chunk_to_entity(self, chunk_id: str, entity_name: str) -> None:
        """Create a MENTIONS relationship from a chunk to an entity.

        Args:
            chunk_id: The chunk identifier.
            entity_name: The mentioned entity's name.
        """
        query = (
            "MATCH (c:Chunk {id: $chunk_id}) "
            "MERGE (e:Entity {name: $entity_name}) "
            "MERGE (c)-[:MENTIONS]->(e)"
        )
        self._run(query, {"chunk_id": chunk_id, "entity_name": entity_name})

    # ------------------------------------------------------------------
    # Document deduplication
    # ------------------------------------------------------------------
    def document_exists(self, doc_id: str) -> bool:
        """Return True if a :Document node with ``doc_id`` already exists.

        Args:
            doc_id: The document identifier to check.
        """
        query = (
            "MATCH (d:Document {id: $doc_id}) "
            "RETURN d.ingested AS ingested LIMIT 1"
        )
        rows = self._run(query, {"doc_id": doc_id})
        return bool(rows) and rows[0].get("ingested") is True

    def mark_document_ingested(self, doc_id: str) -> None:
        """Mark a document as fully ingested for deduplication purposes.

        Args:
            doc_id: The document identifier.
        """
        query = (
            "MERGE (d:Document {id: $doc_id}) SET d.ingested = true"
        )
        self._run(query, {"doc_id": doc_id})

    def delete_document(
        self, doc_id: str, prune_entities: bool = True
    ) -> dict[str, int]:
        """Delete a document, its chunks and (optionally) orphaned entities.

        Removes the :Document node and every :Chunk it owns. Because entities
        are shared across documents (MERGE'd by name), an entity is only
        removed when pruning is enabled *and* no surviving chunk still
        MENTIONS it. Safe to call on a ``doc_id`` that does not exist.

        Args:
            doc_id: The document identifier to remove.
            prune_entities: If True, also delete entities left with no
                remaining MENTIONS after the chunks are gone.

        Returns:
            A dict ``{chunks_deleted, entities_deleted}``.
        """
        chunk_rows = self._run(
            "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) "
            "DETACH DELETE c "
            "RETURN count(c) AS n",
            {"doc_id": doc_id},
        )
        chunks_deleted = chunk_rows[0]["n"] if chunk_rows else 0
        self._run("MATCH (d:Document {id: $doc_id}) DETACH DELETE d", {"doc_id": doc_id})

        entities_deleted = 0
        if prune_entities:
            entity_rows = self._run(
                "MATCH (e:Entity) WHERE NOT (:Chunk)-[:MENTIONS]->(e) "
                "DETACH DELETE e RETURN count(e) AS n"
            )
            entities_deleted = entity_rows[0]["n"] if entity_rows else 0

        logger.info(
            f"Deleted document {doc_id}: {chunks_deleted} chunks, "
            f"{entities_deleted} orphan entities"
        )
        return {"chunks_deleted": chunks_deleted, "entities_deleted": entities_deleted}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def vector_search(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Return the ``top_k`` most similar chunks to the query embedding.

        Args:
            embedding: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            A list of dicts with ``chunk_id``, ``text``, ``doc_id`` and ``score``.
        """
        query = (
            "CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding) "
            "YIELD node, score "
            "RETURN node.id AS chunk_id, node.text AS text, "
            "node.doc_id AS doc_id, score "
            "ORDER BY score DESC"
        )
        rows = self._run(query, {"top_k": top_k, "embedding": embedding})
        return [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "doc_id": r["doc_id"],
                "score": r["score"],
            }
            for r in rows
        ]

    def graph_search(
        self, entity_names: list[str], depth: int = 2
    ) -> dict[str, Any]:
        """Traverse the graph from the given entities and collect context.

        Matches entities by name, traverses up to ``depth`` RELATES_TO hops and
        collects chunks that MENTION any entity reached along the way.

        Args:
            entity_names: Seed entity names extracted from the question.
            depth: Maximum number of relationship hops to traverse.

        Returns:
            A dict with ``chunks``, ``paths`` and ``entities_found``.
        """
        if not entity_names:
            return {"chunks": [], "paths": [], "entities_found": []}

        # Collect connected entities and relationship paths.
        path_query = (
            f"MATCH (seed:Entity) WHERE seed.name IN $names "
            f"MATCH path = (seed)-[r:RELATES_TO*1..{depth}]-(other:Entity) "
            "WITH seed, other, r "
            "UNWIND r AS rel "
            "WITH startNode(rel) AS a, endNode(rel) AS b, rel "
            "RETURN DISTINCT a.name AS source, rel.relation AS relation, "
            "b.name AS target, rel.description AS description"
        )
        path_rows = self._run(path_query, {"names": entity_names})
        paths = [
            {
                "source": r["source"],
                "relation": r["relation"],
                "target": r["target"],
                "description": r["description"],
            }
            for r in path_rows
        ]

        # Collect chunks mentioning the seed entities or their neighbours.
        chunk_query = (
            f"MATCH (seed:Entity) WHERE seed.name IN $names "
            f"MATCH (seed)-[:RELATES_TO*0..{depth}]-(reached:Entity) "
            "MATCH (c:Chunk)-[:MENTIONS]->(reached) "
            "RETURN DISTINCT c.id AS chunk_id, c.text AS text, "
            "c.doc_id AS doc_id"
        )
        chunk_rows = self._run(chunk_query, {"names": entity_names})
        chunks = [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "doc_id": r["doc_id"],
                "score": None,
            }
            for r in chunk_rows
        ]

        # Which of the requested seed entities actually exist in the graph.
        found_query = (
            "MATCH (e:Entity) WHERE e.name IN $names RETURN e.name AS name"
        )
        found_rows = self._run(found_query, {"names": entity_names})
        entities_found = [r["name"] for r in found_rows]

        return {"chunks": chunks, "paths": paths, "entities_found": entities_found}

    def get_entity_neighbors(self, name: str, depth: int = 2) -> dict[str, Any]:
        """Return the subgraph surrounding a single entity.

        Args:
            name: The centre entity name.
            depth: Number of hops to include.

        Returns:
            A dict with ``entity``, ``neighbors`` and ``relationships``.
        """
        entity_query = (
            "MATCH (e:Entity {name: $name}) "
            "RETURN e.name AS name, e.type AS type, e.description AS description"
        )
        entity_rows = self._run(entity_query, {"name": name})
        if not entity_rows:
            return {"entity": None, "neighbors": [], "relationships": []}
        entity = entity_rows[0]

        neighbor_query = (
            f"MATCH (e:Entity {{name: $name}})-[:RELATES_TO*1..{depth}]-(n:Entity) "
            "WHERE n.name <> $name "
            "RETURN DISTINCT n.name AS name, n.type AS type, "
            "n.description AS description"
        )
        neighbor_rows = self._run(neighbor_query, {"name": name})
        neighbors = [
            {
                "name": r["name"],
                "type": r["type"] or "CONCEPT",
                "description": r["description"] or "",
            }
            for r in neighbor_rows
        ]

        rel_query = (
            f"MATCH (e:Entity {{name: $name}})-[r:RELATES_TO*1..{depth}]-(:Entity) "
            "UNWIND r AS rel "
            "WITH startNode(rel) AS a, endNode(rel) AS b, rel "
            "RETURN DISTINCT a.name AS source, rel.relation AS relation, "
            "b.name AS target, rel.description AS description"
        )
        rel_rows = self._run(rel_query, {"name": name})
        relationships = [
            {
                "source": r["source"],
                "relation": r["relation"],
                "target": r["target"],
                "description": r["description"] or "",
            }
            for r in rel_rows
        ]

        return {
            "entity": {
                "name": entity["name"],
                "type": entity["type"] or "CONCEPT",
                "description": entity["description"] or "",
            },
            "neighbors": neighbors,
            "relationships": relationships,
        }

    def get_all_entities(
        self, type_filter: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List entities, optionally filtered by type.

        Args:
            type_filter: Restrict to a specific entity type, or ``None`` for all.
            limit: Maximum number of entities to return.

        Returns:
            A list of entity dicts.
        """
        if type_filter:
            query = (
                "MATCH (e:Entity) WHERE e.type = $type "
                "RETURN e.name AS name, e.type AS type, "
                "e.description AS description "
                "ORDER BY e.name LIMIT $limit"
            )
            params: dict[str, Any] = {"type": type_filter, "limit": limit}
        else:
            query = (
                "MATCH (e:Entity) "
                "RETURN e.name AS name, e.type AS type, "
                "e.description AS description "
                "ORDER BY e.name LIMIT $limit"
            )
            params = {"limit": limit}
        rows = self._run(query, params)
        return [
            {
                "name": r["name"],
                "type": r["type"] or "CONCEPT",
                "description": r["description"] or "",
            }
            for r in rows
        ]

    def find_relation_paths(self, source: str, target: str) -> list[dict[str, Any]]:
        """Return the shortest RELATES_TO path between two entities.

        Args:
            source: Source entity name.
            target: Target entity name.

        Returns:
            A list of relationship dicts describing the path, empty if none.
        """
        query = (
            "MATCH (a:Entity {name: $source}), (b:Entity {name: $target}), "
            "p = shortestPath((a)-[:RELATES_TO*1..10]-(b)) "
            "UNWIND relationships(p) AS rel "
            "WITH startNode(rel) AS s, endNode(rel) AS t, rel "
            "RETURN s.name AS source, rel.relation AS relation, "
            "t.name AS target, rel.description AS description"
        )
        rows = self._run(query, {"source": source, "target": target})
        return [
            {
                "source": r["source"],
                "relation": r["relation"],
                "target": r["target"],
                "description": r["description"] or "",
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def get_documents(self) -> list[dict[str, Any]]:
        """List ingested documents with chunk and entity counts.

        Returns:
            A list of dicts ``{doc_id, file_name, chunk_count, entity_count,
            ingested}`` ordered by file name.
        """
        query = (
            "MATCH (d:Document) "
            "OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk) "
            "OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity) "
            "RETURN d.id AS doc_id, coalesce(d.file_name, '') AS file_name, "
            "d.ingested AS ingested, count(DISTINCT c) AS chunk_count, "
            "count(DISTINCT e) AS entity_count "
            "ORDER BY file_name, doc_id"
        )
        rows = self._run(query)
        return [
            {
                "doc_id": r["doc_id"],
                "file_name": r["file_name"] or r["doc_id"],
                "ingested": bool(r["ingested"]),
                "chunk_count": r["chunk_count"],
                "entity_count": r["entity_count"],
            }
            for r in rows
        ]

    def get_document_graph(self, doc_id: str) -> dict[str, Any]:
        """Return the full chunk/entity structure of one document.

        Includes each chunk's embedding vector and the entities it mentions,
        plus the RELATES_TO relationships among those entities. Used by the
        pipeline visualiser.

        Args:
            doc_id: The document identifier.

        Returns:
            A dict with ``chunks`` (each with ``embedding`` and ``entities``),
            ``entities`` (name + type for colouring) and ``relationships``.
        """
        chunk_query = (
            "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) "
            "OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity) "
            "RETURN c.id AS chunk_id, c.chunk_index AS chunk_index, "
            "coalesce(c.page_num, 0) AS page_num, c.text AS text, "
            "coalesce(c.block_type, 'text') AS block_type, "
            "coalesce(c.section, '') AS section, "
            "c.embedding AS embedding, "
            "collect(DISTINCT e.name) AS entities "
            "ORDER BY chunk_index"
        )
        chunk_rows = self._run(chunk_query, {"doc_id": doc_id})
        chunks = [
            {
                "chunk_id": r["chunk_id"],
                "chunk_index": r["chunk_index"],
                "page_num": r["page_num"],
                "text": r["text"],
                "block_type": r["block_type"] or "text",
                "section": r["section"] or "",
                "embedding": r["embedding"] or [],
                "entities": [e for e in r["entities"] if e],
            }
            for r in chunk_rows
        ]

        rel_query = (
            "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(:Chunk)"
            "-[:MENTIONS]->(a:Entity) "
            "MATCH (a)-[r:RELATES_TO]->(b:Entity) "
            "WHERE (d)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(b) "
            "RETURN DISTINCT a.name AS source, r.relation AS relation, "
            "b.name AS target, coalesce(r.description, '') AS description"
        )
        rel_rows = self._run(rel_query, {"doc_id": doc_id})
        relationships = [
            {
                "source": r["source"],
                "relation": r["relation"],
                "target": r["target"],
                "description": r["description"],
            }
            for r in rel_rows
        ]

        entity_query = (
            "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(:Chunk)"
            "-[:MENTIONS]->(e:Entity) "
            "RETURN DISTINCT e.name AS name, coalesce(e.type, 'CONCEPT') AS type"
        )
        entity_rows = self._run(entity_query, {"doc_id": doc_id})
        entities = [
            {"name": r["name"], "type": r["type"] or "CONCEPT"}
            for r in entity_rows
        ]

        return {
            "chunks": chunks,
            "entities": entities,
            "relationships": relationships,
        }

    # ------------------------------------------------------------------
    # Lifecycle / health
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if the database answers a trivial query."""
        try:
            rows = self._run("RETURN 1 AS ok")
            return bool(rows) and rows[0].get("ok") == 1
        except Neo4jError:
            return False

    def close(self) -> None:
        """Close the underlying driver connection pool."""
        self._driver.close()
        logger.info("Neo4j driver closed")
