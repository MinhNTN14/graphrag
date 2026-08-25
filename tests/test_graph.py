"""Integration tests for the Neo4j client.

These tests require a live Neo4j instance and are skipped when ``NEO4J_URI``
is not set in the environment. They exercise real Cypher against the database.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("NEO4J_URI"), reason="Neo4j not available"
)


@pytest.fixture(scope="module")
def client():
    """Provide a connected Neo4jClient and close it after the module runs."""
    from app.core.graph.neo4j_client import Neo4jClient

    c = Neo4jClient()
    yield c
    c.close()


def _rand(prefix: str) -> str:
    """Return a unique name with the given prefix for test isolation."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_upsert_entity(client) -> None:
    """An upserted entity is returned by get_all_entities."""
    name = _rand("Entity")
    client.upsert_entity(name, "TECHNOLOGY", "A test entity.")
    entities = client.get_all_entities(type_filter="TECHNOLOGY", limit=200)
    names = [e["name"] for e in entities]
    assert name in names


def test_upsert_relationship(client) -> None:
    """An upserted relationship can be found via find_relation_paths."""
    a = _rand("A")
    b = _rand("B")
    client.upsert_entity(a, "CONCEPT", "A")
    client.upsert_entity(b, "CONCEPT", "B")
    client.upsert_relationship(a, "RELATES", b, "a relates to b")
    path = client.find_relation_paths(a, b)
    assert any(r["source"] == a and r["target"] == b for r in path)


def test_upsert_chunk_with_embedding(client) -> None:
    """A chunk with an embedding is retrievable via vector search."""
    chunk_id = _rand("chunk")
    embedding = [0.01] * 768
    client.upsert_chunk(chunk_id, "vector search test", embedding, "docX", 0)
    results = client.vector_search(embedding, top_k=5)
    ids = [r["chunk_id"] for r in results]
    assert chunk_id in ids


def test_vector_search_returns_topk(client) -> None:
    """Vector search returns at most top_k results."""
    embedding = [0.02] * 768
    for i in range(6):
        client.upsert_chunk(_rand(f"vs{i}"), f"text {i}", embedding, "docV", i)
    results = client.vector_search(embedding, top_k=3)
    assert len(results) <= 3


def test_graph_search_traversal(client) -> None:
    """Graph search finds chunks mentioning a seed entity's neighbours."""
    a = _rand("GA")
    b = _rand("GB")
    client.upsert_entity(a, "CONCEPT", "A")
    client.upsert_entity(b, "CONCEPT", "B")
    client.upsert_relationship(a, "LINKS", b, "a links b")

    chunk_id = _rand("gchunk")
    client.upsert_chunk(chunk_id, "graph traversal text", [0.03] * 768, "docG", 0)
    client.link_chunk_to_entity(chunk_id, b)

    result = client.graph_search([a], depth=2)
    chunk_ids = [c["chunk_id"] for c in result["chunks"]]
    assert chunk_id in chunk_ids
    assert a in result["entities_found"]
