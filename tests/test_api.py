"""API tests using FastAPI's TestClient with mocked Neo4j and Gemini.

All external collaborators (Neo4j driver, Gemini embedder/extractor/generator)
are patched so the tests run fully offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Build a TestClient with every external dependency mocked out."""
    neo4j_instance = MagicMock()
    neo4j_instance.health_check.return_value = True
    neo4j_instance.document_exists.return_value = False
    neo4j_instance.get_all_entities.return_value = [
        {"name": "Neo4j", "type": "TECHNOLOGY", "description": "A graph DB."}
    ]

    embedder_instance = MagicMock()
    embedder_instance.embed.return_value = [0.0] * 768

    extractor_instance = MagicMock()
    extractor_instance.extract.return_value = {"entities": [], "relationships": []}

    generator_instance = MagicMock()
    generator_instance.generate.return_value = {
        "answer": "This is a test answer.",
        "sources": [
            {
                "chunk_id": "doc_0",
                "text": "chunk text",
                "doc_id": "doc",
                "score": 0.9,
                "source": "vector",
            }
        ],
        "graph_context": [],
        "mode_used": "hybrid",
        "entities_found": [],
    }

    with patch("app.main.Neo4jClient", return_value=neo4j_instance), patch(
        "app.main.GeminiEmbedder", return_value=embedder_instance
    ), patch(
        "app.main.EntityExtractor", return_value=extractor_instance
    ), patch(
        "app.main.AnswerGenerator", return_value=generator_instance
    ):
        from app.main import app

        with TestClient(app) as test_client:
            # Ensure the hybrid retriever delegates to our mocked generator by
            # returning a simple retrieval result regardless of Neo4j content.
            app.state.hybrid_retriever = MagicMock()
            app.state.hybrid_retriever.retrieve.return_value = {
                "chunks": [],
                "graph_context": [],
                "mode_used": "hybrid",
                "entities_found": [],
            }
            app.state.answer_generator = generator_instance
            app.state.neo4j_client = neo4j_instance
            yield test_client


def test_health_endpoint(client) -> None:
    """The health endpoint reports ok and Neo4j connectivity."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["neo4j"] is True


def test_ingest_txt_file(client) -> None:
    """Uploading a .txt file returns ingestion statistics."""
    files = {"file": ("sample.txt", b"GraphRAG is a system. It uses Neo4j.", "text/plain")}
    resp = client.post("/api/v1/ingest", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert "doc_id" in body
    assert body["chunks_processed"] >= 1


def test_query_hybrid_mode(client) -> None:
    """A hybrid-mode query returns an answer and sources."""
    payload = {"question": "What is GraphRAG?", "mode": "hybrid", "top_k": 5}
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "This is a test answer."
    assert body["mode_used"] == "hybrid"
    assert len(body["sources"]) == 1


def test_query_invalid_mode(client) -> None:
    """An invalid mode value is rejected with HTTP 422."""
    payload = {"question": "hi", "mode": "invalid", "top_k": 5}
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 422


def test_graph_entities_endpoint(client) -> None:
    """The graph entities endpoint returns the mocked entity list."""
    resp = client.get("/api/v1/graph/entities?type=TECHNOLOGY&limit=20")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert body[0]["name"] == "Neo4j"
