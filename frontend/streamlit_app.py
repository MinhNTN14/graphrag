"""Streamlit frontend for the GraphRAG application.

Provides two tabs:
  1. Document Q&A  — upload documents and ask questions.
  2. Knowledge Graph Explorer — browse entities and their relationships.
"""

from __future__ import annotations

from typing import Any

import json

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import Config, Edge, Node, agraph

st.set_page_config(page_title="GraphRAG", page_icon="🕸️", layout="wide")

ENTITY_TYPES = [
    "ALL",
    "PERSON",
    "ORGANIZATION",
    "CONCEPT",
    "TECHNOLOGY",
    "LOCATION",
    "EVENT",
]

# Shared colour language across every graph in the app.
DOC_COLOR = "#8E44AD"
CHUNK_COLOR = "#2E86C1"
QUERY_COLOR = "#0D7377"

# Distinct look for non-prose chunks so tables/images/code stand out.
BLOCK_STYLE = {
    "text": (CHUNK_COLOR, "dot", "Chunk (text)"),
    "table": ("#E67E22", "square", "Chunk (table)"),
    "image": ("#16A085", "triangle", "Chunk (image)"),
    "code": ("#7F8C8D", "square", "Chunk (code)"),
}
RETRIEVED_COLOR = "#C3423F"
DIMMED_COLOR = "#B4C4CC"
ENTITY_COLORS = {
    "PERSON": "#E76F51",
    "ORGANIZATION": "#E9C46A",
    "TECHNOLOGY": "#2A9D8F",
    "CONCEPT": "#4C6EF5",
    "LOCATION": "#8AB17D",
    "EVENT": "#9B5DE5",
}
DEFAULT_ENTITY_COLOR = "#6C757D"


def _entity_color(entity_type: str | None) -> str:
    """Return the palette colour for an entity type."""
    return ENTITY_COLORS.get((entity_type or "").upper(), DEFAULT_ENTITY_COLOR)


def _legend_html(items: list[tuple[str, str]]) -> str:
    """Build an inline HTML legend from ``(color, label)`` pairs."""
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;'
        f'font-size:0.85rem;white-space:nowrap;">'
        f'<span style="width:12px;height:12px;border-radius:50%;'
        f'background:{color};display:inline-block;margin-right:5px;'
        f'border:1px solid rgba(0,0,0,0.15);"></span>{label}</span>'
        for color, label in items
    )
    return f'<div style="margin:2px 0 10px 0;line-height:1.9;">{chips}</div>'


# ----------------------------------------------------------------------
# API helpers
# ----------------------------------------------------------------------
def _api_base() -> str:
    """Return the configured API base URL from the sidebar."""
    return st.session_state.get("api_url", "http://localhost:8000").rstrip("/")


def ingest_file(uploaded_file: Any) -> dict[str, Any]:
    """Upload a document to the ingestion endpoint.

    Args:
        uploaded_file: The Streamlit uploaded file object.

    Returns:
        The JSON response body.
    """
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    response = requests.post(
        f"{_api_base()}/api/v1/ingest", files=files, timeout=600
    )
    response.raise_for_status()
    return response.json()


def run_query(question: str, mode: str, top_k: int) -> dict[str, Any]:
    """Send a question to the query endpoint.

    Args:
        question: The user's question.
        mode: Retrieval mode ('hybrid', 'vector', 'graph').
        top_k: Number of chunks to retrieve.

    Returns:
        The JSON response body.
    """
    payload = {"question": question, "mode": mode, "top_k": top_k}
    response = requests.post(
        f"{_api_base()}/api/v1/query", json=payload, timeout=600
    )
    response.raise_for_status()
    return response.json()


def fetch_entities(type_filter: str, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch entities from the graph endpoint.

    Args:
        type_filter: Entity type filter, or 'ALL' for no filter.
        limit: Maximum number of entities.

    Returns:
        A list of entity dicts.
    """
    params: dict[str, Any] = {"limit": limit}
    if type_filter != "ALL":
        params["type"] = type_filter
    response = requests.get(
        f"{_api_base()}/api/v1/graph/entities", params=params, timeout=60
    )
    response.raise_for_status()
    return response.json()


def fetch_subgraph(name: str, depth: int = 2) -> dict[str, Any]:
    """Fetch the subgraph around an entity.

    Args:
        name: The entity name.
        depth: Traversal depth.

    Returns:
        The subgraph JSON body.
    """
    response = requests.get(
        f"{_api_base()}/api/v1/graph/entity/{name}/neighbors",
        params={"depth": depth},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def fetch_documents() -> list[dict[str, Any]]:
    """Fetch the list of ingested documents.

    Returns:
        A list of document summary dicts.
    """
    response = requests.get(f"{_api_base()}/api/v1/graph/documents", timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_document_graph(doc_id: str) -> dict[str, Any]:
    """Fetch a document's chunks (with embeddings) and relationships.

    Args:
        doc_id: The document identifier.

    Returns:
        A dict with ``chunks`` and ``relationships``.
    """
    response = requests.get(
        f"{_api_base()}/api/v1/graph/document/{doc_id}", timeout=120
    )
    response.raise_for_status()
    return response.json()


def embed_query_api(question: str, top_k: int) -> dict[str, Any]:
    """Embed a query and return its vector plus nearest chunk ids.

    Args:
        question: The user's question.
        top_k: Number of neighbours to retrieve.

    Returns:
        A dict ``{embedding, retrieved}``.
    """
    response = requests.post(
        f"{_api_base()}/api/v1/graph/embed-query",
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def pca_2d(matrix: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project high-dimensional vectors to 2D via PCA (numpy SVD).

    Args:
        matrix: A list of equal-length embedding vectors.

    Returns:
        A tuple ``(coords, mean, components)`` where ``coords`` is an (n, 2)
        array, ``mean`` is the (d,) column mean, and ``components`` is the
        (2, d) projection matrix (so a new vector v maps to
        ``(v - mean) @ components.T``).
    """
    matrix_arr = np.asarray(matrix, dtype=float)
    mean = matrix_arr.mean(axis=0)
    centered = matrix_arr - mean
    if centered.shape[0] < 2:
        # Not enough points for a meaningful projection.
        components = np.eye(2, centered.shape[1])
        return np.zeros((centered.shape[0], 2)), mean, components
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    coords = centered @ components.T
    return coords, mean, components


def knn_edges(
    embeddings: list[list[float]], coords: np.ndarray, k: int = 2
) -> list[dict[str, float]]:
    """Build faint links between each chunk and its ``k`` nearest neighbours.

    Uses cosine similarity in the original high-dimensional space (what the
    retriever actually uses), but draws the links in the 2D projection so the
    scatter reads as a real "semantic neighbourhood" map rather than loose dots.

    Args:
        embeddings: The chunk embedding vectors.
        coords: The (n, 2) PCA projection of those vectors.
        k: Number of neighbours to connect per chunk.

    Returns:
        A list of ``{x, y, x2, y2, sim}`` line-segment rows.
    """
    mat = np.asarray(embeddings, dtype=float)
    n = mat.shape[0]
    if n < 2:
        return []
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / np.clip(norms, 1e-12, None)
    sims = normed @ normed.T
    np.fill_diagonal(sims, -np.inf)
    rows: list[dict[str, float]] = []
    seen: set[tuple[int, int]] = set()
    kk = min(k, n - 1)
    for i in range(n):
        for j in np.argsort(sims[i])[-kk:]:
            pair = (min(i, int(j)), max(i, int(j)))
            if pair in seen:
                continue
            seen.add(pair)
            rows.append(
                {
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "x2": float(coords[int(j), 0]),
                    "y2": float(coords[int(j), 1]),
                    "sim": round(float(sims[i, int(j)]), 3),
                }
            )
    return rows


def fetch_relation_path(source: str, target: str) -> list[dict[str, Any]]:
    """Fetch the shortest relationship path between two entities.

    Args:
        source: Source entity name.
        target: Target entity name.

    Returns:
        A list of relationship dicts.
    """
    response = requests.get(
        f"{_api_base()}/api/v1/graph/relations",
        params={"source": source, "target": target},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def render_sidebar() -> tuple[str, int]:
    """Render the sidebar controls.

    Returns:
        A tuple ``(mode, top_k)`` selected by the user.
    """
    st.sidebar.title("⚙️ Settings")
    st.session_state["api_url"] = st.sidebar.text_input(
        "API URL", value=st.session_state.get("api_url", "http://localhost:8000")
    )
    mode = st.sidebar.selectbox(
        "Retrieval mode", ["hybrid", "vector", "graph"], index=0
    )
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=5)

    st.sidebar.markdown("---")
    if st.sidebar.button("Check API health"):
        try:
            resp = requests.get(f"{_api_base()}/health", timeout=10)
            st.sidebar.success(f"Health: {resp.json()}")
        except requests.RequestException as exc:
            st.sidebar.error(f"Unreachable: {exc}")
    return mode, top_k


# ----------------------------------------------------------------------
# Tab 1: Document Q&A
# ----------------------------------------------------------------------
def render_qa_tab(mode: str, top_k: int) -> None:
    """Render the Document Q&A tab.

    Args:
        mode: Selected retrieval mode.
        top_k: Selected top-k value.
    """
    st.header("📄 Document Q&A")

    uploaded = st.file_uploader(
        "Upload a document", type=["pdf", "txt", "md"], key="qa_uploader"
    )
    if uploaded is not None and st.button("Ingest Document"):
        with st.spinner("Ingesting document..."):
            try:
                result = ingest_file(uploaded)
                st.success(result.get("message", "Ingested."))
                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks", result.get("chunks_processed", 0))
                col2.metric("Entities", result.get("entities_created", 0))
                col3.metric("Relationships", result.get("relationships_created", 0))
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")

    st.markdown("---")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Render existing history.
    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            _render_sources(turn.get("sources", []))
            _render_graph_context(turn.get("graph_context", []), turn.get("mode_used"))

    question = st.chat_input("Ask a question about your documents")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = run_query(question, mode, top_k)
                except requests.RequestException as exc:
                    st.error(f"Query failed: {exc}")
                    return
            st.write(result["answer"])
            _render_sources(result.get("sources", []))
            _render_graph_context(
                result.get("graph_context", []), result.get("mode_used")
            )
        st.session_state["chat_history"].append(
            {
                "question": question,
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "graph_context": result.get("graph_context", []),
                "mode_used": result.get("mode_used"),
            }
        )


def _render_sources(sources: list[dict[str, Any]]) -> None:
    """Render an expandable list of source chunks.

    Args:
        sources: Source dicts from a query response.
    """
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for src in sources:
            score = src.get("score")
            score_str = f" | score={score:.4f}" if isinstance(score, (int, float)) else ""
            st.markdown(
                f"**{src.get('chunk_id', '?')}** "
                f"(doc `{src.get('doc_id', '?')}`, "
                f"via {src.get('source', '?')}{score_str})"
            )
            st.caption(src.get("text", ""))
            st.markdown("---")


def _render_graph_context(
    graph_context: list[dict[str, Any]], mode_used: str | None
) -> None:
    """Render an expandable list of relationship paths.

    Args:
        graph_context: Relationship path dicts.
        mode_used: The retrieval mode used.
    """
    if mode_used not in ("hybrid", "graph") or not graph_context:
        return
    with st.expander(f"🕸️ Graph Context ({len(graph_context)} relationships)"):
        for rel in graph_context:
            st.markdown(
                f"`{rel.get('source', '?')}` "
                f"**-[{rel.get('relation', 'RELATED')}]->** "
                f"`{rel.get('target', '?')}`"
            )
            if rel.get("description"):
                st.caption(rel["description"])


# ----------------------------------------------------------------------
# Tab 2: Knowledge Graph Explorer
# ----------------------------------------------------------------------
def render_graph_tab() -> None:
    """Render the Knowledge Graph Explorer tab."""
    st.header("🕸️ Knowledge Graph Explorer")

    col1, col2 = st.columns([2, 1])
    with col1:
        search_text = st.text_input("Filter by name (optional)", key="entity_search")
    with col2:
        type_filter = st.selectbox("Type", ENTITY_TYPES, index=0)

    if st.button("Search Entities"):
        _search_and_store_entities(type_filter, search_text)

    entities = st.session_state.get("entities", [])
    if entities:
        st.subheader("Entities")
        st.dataframe(
            [
                {"Name": e["name"], "Type": e["type"], "Description": e["description"]}
                for e in entities
            ],
            use_container_width=True,
        )
        names = [e["name"] for e in entities]
        selected = st.selectbox("Inspect entity subgraph", names)
        depth = st.slider("Depth", min_value=1, max_value=4, value=2, key="subgraph_depth")
        if st.button("Load Subgraph"):
            _render_subgraph(selected, depth)

    st.markdown("---")
    st.subheader("🔗 Relation Finder")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        entity_a = st.text_input("Entity A")
    with rcol2:
        entity_b = st.text_input("Entity B")
    if st.button("Find Path"):
        _render_relation_path(entity_a, entity_b)


def _search_and_store_entities(type_filter: str, search_text: str) -> None:
    """Fetch entities and store the (optionally name-filtered) result.

    Args:
        type_filter: The entity type filter.
        search_text: Optional case-insensitive name substring filter.
    """
    with st.spinner("Searching entities..."):
        try:
            entities = fetch_entities(type_filter, limit=100)
        except requests.RequestException as exc:
            st.error(f"Search failed: {exc}")
            return
    if search_text:
        lowered = search_text.lower()
        entities = [e for e in entities if lowered in e["name"].lower()]
    st.session_state["entities"] = entities
    if not entities:
        st.info("No entities found.")


def _render_subgraph(name: str, depth: int) -> None:
    """Render the subgraph visualisation and relationship table for an entity.

    Args:
        name: The centre entity name.
        depth: Traversal depth.
    """
    with st.spinner(f"Loading subgraph for {name}..."):
        try:
            subgraph = fetch_subgraph(name, depth)
        except requests.RequestException as exc:
            st.error(f"Failed to load subgraph: {exc}")
            return

    centre = subgraph["entity"]
    neighbors = subgraph["neighbors"]
    relationships = subgraph["relationships"]

    # Build node set: centre + neighbours.
    node_ids = {centre["name"]}
    nodes = [
        Node(id=centre["name"], label=centre["name"], size=28, color="#C3423F")
    ]
    for neighbour in neighbors:
        if neighbour["name"] not in node_ids:
            node_ids.add(neighbour["name"])
            nodes.append(
                Node(
                    id=neighbour["name"],
                    label=neighbour["name"],
                    size=20,
                    color="#0D7377",
                )
            )

    edges = []
    for rel in relationships:
        # Only draw edges whose endpoints are present as nodes.
        for endpoint in (rel["source"], rel["target"]):
            if endpoint not in node_ids:
                node_ids.add(endpoint)
                nodes.append(
                    Node(id=endpoint, label=endpoint, size=20, color="#0D7377")
                )
        edges.append(
            Edge(source=rel["source"], target=rel["target"], label=rel["relation"])
        )

    config = Config(width=700, height=400, directed=True)
    agraph(nodes=nodes, edges=edges, config=config)

    if relationships:
        st.subheader("Relationships")
        st.dataframe(
            [
                {
                    "Source": r["source"],
                    "Relation": r["relation"],
                    "Target": r["target"],
                    "Description": r["description"],
                }
                for r in relationships
            ],
            use_container_width=True,
        )

    st.subheader("Cypher used")
    st.code(
        "MATCH (e:Entity {name: $name})-[:RELATES_TO*1.."
        f"{depth}]-(n:Entity)\n"
        "WHERE n.name <> $name\n"
        "RETURN DISTINCT n.name, n.type, n.description",
        language="cypher",
    )


def _render_relation_path(entity_a: str, entity_b: str) -> None:
    """Render the shortest relationship path between two entities.

    Args:
        entity_a: Source entity name.
        entity_b: Target entity name.
    """
    if not entity_a or not entity_b:
        st.warning("Please provide both entities.")
        return
    with st.spinner("Finding path..."):
        try:
            relations = fetch_relation_path(entity_a, entity_b)
        except requests.RequestException as exc:
            st.error(f"Failed: {exc}")
            return
    if not relations:
        st.info("No path found between these entities.")
        return
    st.dataframe(
        [
            {
                "Source": r["source"],
                "Relation": r["relation"],
                "Target": r["target"],
                "Description": r["description"],
            }
            for r in relations
        ],
        use_container_width=True,
    )
    st.code(
        "MATCH (a:Entity {name: $source}), (b:Entity {name: $target}),\n"
        "  p = shortestPath((a)-[:RELATES_TO*1..10]-(b))\n"
        "RETURN p",
        language="cypher",
    )


# ----------------------------------------------------------------------
# Tab 3: Pipeline Visualizer
# ----------------------------------------------------------------------
def render_pipeline_tab() -> None:
    """Render the Pipeline Visualizer tab.

    Shows, for a selected ingested document: the Document→Chunk→Entity graph,
    the chunk embeddings projected to a 2D "vector space", and where a user
    query lands in that space along with what it retrieves.
    """
    st.header("🔬 Pipeline Visualizer")
    st.caption(
        "See how an uploaded file becomes chunks + a knowledge graph + vectors, "
        "and how a query retrieves them."
    )

    try:
        documents = fetch_documents()
    except requests.RequestException as exc:
        st.error(f"Could not load documents: {exc}")
        return
    if not documents:
        st.info("No documents ingested yet. Upload a file in the Document Q&A tab first.")
        return

    labels = {
        f"{d['file_name']}  —  {d['chunk_count']} chunks, "
        f"{d['entity_count']} entities": d
        for d in documents
    }
    choice = st.selectbox("Select an ingested document", list(labels.keys()))
    doc = labels[choice]

    with st.spinner("Loading document graph + embeddings..."):
        try:
            graph = fetch_document_graph(doc["doc_id"])
        except requests.RequestException as exc:
            st.error(f"Failed to load document: {exc}")
            return

    chunks = [c for c in graph["chunks"] if c.get("embedding")]
    relationships = graph["relationships"]
    if not chunks:
        st.warning(
            "This document has no embedded chunks yet (ingest may still be running)."
        )
        return

    entity_types = {
        e["name"]: e.get("type", "CONCEPT") for e in graph.get("entities", [])
    }
    all_entities = sorted({e for c in chunks for e in c["entities"]})
    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks", len(chunks))
    m2.metric("Entities", len(all_entities))
    m3.metric("Relationships", len(relationships))

    # --- Section 1: ingestion graph -------------------------------------
    st.subheader("1️⃣ Ingestion graph — Document → Chunks → Entities")
    st.caption(
        "The knowledge graph built from this file. Node size grows with how "
        "connected a node is; entities are coloured by type. Click a node to "
        "highlight its neighbourhood."
    )
    show_entities = st.checkbox(
        "Show entity nodes (MENTIONS + RELATES_TO)", value=len(all_entities) <= 120
    )
    _render_ingestion_graph(doc, chunks, relationships, entity_types, show_entities)

    # --- Section 2: embedding space -------------------------------------
    st.subheader("2️⃣ Embedding space — each paragraph as a vector")
    dims = len(chunks[0]["embedding"])
    st.caption(
        f"Each chunk's {dims}-dim embedding projected to 2D via PCA. "
        "Nearby points are semantically similar paragraphs — this is the "
        "'vector database' the retriever searches."
    )
    embeddings = [c["embedding"] for c in chunks]
    coords, mean, comps = pca_2d(embeddings)
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "chunk": [f"#{c['chunk_index']}" for c in chunks],
            "page": [c["page_num"] for c in chunks],
            "n_entities": [len(c["entities"]) for c in chunks],
            "preview": [c["text"][:140].replace("\n", " ") + "…" for c in chunks],
        }
    )
    show_links = st.checkbox(
        "Link each paragraph to its nearest neighbours (cosine similarity)",
        value=True,
        key="viz_knn",
    )

    layers: list[alt.Chart] = []
    if show_links:
        link_df = pd.DataFrame(knn_edges(embeddings, coords, k=2))
        if not link_df.empty:
            layers.append(
                alt.Chart(link_df)
                .mark_rule(color="#9AA7B0", opacity=0.35)
                .encode(
                    x="x:Q",
                    y="y:Q",
                    x2="x2:Q",
                    y2="y2:Q",
                    strokeWidth=alt.StrokeWidth(
                        "sim:Q", legend=None, scale=alt.Scale(range=[0.4, 3])
                    ),
                    tooltip=["sim"],
                )
            )
    layers.append(
        alt.Chart(df)
        .mark_circle(opacity=0.85, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", title="PC-1 (largest variance direction)"),
            y=alt.Y("y:Q", title="PC-2"),
            size=alt.Size(
                "n_entities:Q", title="# entities", scale=alt.Scale(range=[80, 500])
            ),
            color=alt.Color(
                "page:N", title="Page", scale=alt.Scale(scheme="viridis")
            ),
            tooltip=["chunk", "page", "n_entities", "preview"],
        )
    )
    layers.append(
        alt.Chart(df)
        .mark_text(dy=-14, fontSize=11, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="chunk")
    )
    st.altair_chart(
        alt.layer(*layers).interactive().properties(height=460),
        use_container_width=True,
    )
    with st.expander("📋 Paragraph → vector table"):
        st.dataframe(
            pd.DataFrame(
                {
                    "chunk": df["chunk"],
                    "page": df["page"],
                    "text preview": [c["text"][:90].replace("\n", " ") + "…" for c in chunks],
                    "embedding[:5]": [
                        np.round(np.asarray(c["embedding"][:5]), 4).tolist()
                        for c in chunks
                    ],
                    "2D (PC1, PC2)": [
                        (round(float(x), 3), round(float(y), 3))
                        for x, y in zip(df["x"], df["y"])
                    ],
                }
            ),
            use_container_width=True,
        )

    # --- Section 3: query retrieval -------------------------------------
    st.subheader("3️⃣ Query retrieval — where does a question land?")
    st.caption(
        "Your question is embedded into the same space; the nearest chunks "
        "(cosine similarity) are exactly what vector search retrieves."
    )
    question = st.text_input("Question", key="viz_query")
    top_k = st.slider(
        "Top K", 1, min(10, len(chunks)), min(5, len(chunks)), key="viz_topk"
    )
    if st.button("Embed & Retrieve", key="viz_btn") and question.strip():
        with st.spinner("Embedding query & searching..."):
            try:
                res = embed_query_api(question, top_k)
            except requests.RequestException as exc:
                st.error(f"Failed: {exc}")
                return
        _render_query_projection(res, chunks, df, mean, comps)


def all_entity_names(chunks: list[dict[str, Any]]) -> set[str]:
    """Return the set of all entity names mentioned across ``chunks``."""
    return {e for c in chunks for e in c["entities"]}


def _render_ingestion_graph(
    doc: dict[str, Any],
    chunks: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    entity_types: dict[str, str],
    show_entities: bool,
) -> None:
    """Draw the Document→Chunk→Entity graph with an embedded vis.js network.

    Rendered through a self-contained ``components.html`` iframe (rather than the
    streamlit-agraph component, whose validation makes it blank on custom
    physics). Entities are coloured by type, nodes are sized by degree, and the
    barnesHut ``avoidOverlap`` physics keeps labels from colliding.

    Args:
        doc: The document summary dict.
        chunks: Chunk dicts including their mentioned entity names.
        relationships: RELATES_TO relationships among entities.
        entity_types: Map of entity name → type for colouring.
        show_entities: Whether to include entity nodes and edges.
    """
    # First pass: compute a degree for every node so we can size by importance.
    degree: dict[str, int] = {}

    def bump(*names: str) -> None:
        for name in names:
            degree[name] = degree.get(name, 0) + 1

    doc_id = doc["doc_id"]
    for chunk in chunks:
        bump(doc_id, chunk["chunk_id"])
        if show_entities:
            for ename in chunk["entities"]:
                bump(chunk["chunk_id"], ename)
    if show_entities:
        for rel in relationships:
            bump(rel["source"], rel["target"])

    # Declutter controls: by default only the most-connected entities are
    # labelled (the rest show their name on hover), and edge labels are hidden.
    c1, c2, c3 = st.columns([1, 1, 1.4])
    show_all_labels = c1.checkbox("Label every node", value=False, key="ig_all")
    show_rel_labels = c2.checkbox("Relationship names", value=False, key="ig_rel")
    spacing = c3.slider("Spacing", 90, 320, 180, 10, key="ig_spread")

    # Adaptive label cut-off: keep roughly the top ~15 entities by degree.
    ent_names = all_entity_names(chunks) if show_entities else set()
    ent_degrees = sorted((degree.get(e, 1) for e in ent_names), reverse=True)
    label_cut = ent_degrees[min(14, len(ent_degrees) - 1)] if ent_degrees else 1
    # In a big graph, never label the degree-1 leaves (that's what overlaps).
    if len(ent_degrees) > 20:
        label_cut = max(2, label_cut)

    def labeled(name: str) -> bool:
        return show_all_labels or degree.get(name, 1) >= label_cut

    vnodes: list[dict[str, Any]] = []
    vedges: list[dict[str, Any]] = []
    seen: set[str] = set()

    vnodes.append(
        {
            "id": doc_id,
            "label": f"📄 {doc['file_name'][:24]}",
            "size": 30,
            "color": DOC_COLOR,
            "shape": "hexagon",
            "font": {"color": "#4A235A", "size": 16, "bold": True},
            "title": f"Document: {doc['file_name']}",
        }
    )
    seen.add(doc_id)

    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid not in seen:
            btype = chunk.get("block_type", "text")
            color, shape, _ = BLOCK_STYLE.get(btype, BLOCK_STYLE["text"])
            sec = chunk.get("section", "")
            tag = "" if btype == "text" else f" [{btype.upper()}]"
            title = (f"section: {sec}\n" if sec else "") + chunk["text"][:200].replace(
                "\n", " "
            ) + "…"
            vnodes.append(
                {
                    "id": cid,
                    "label": f"#{chunk['chunk_index']}{tag}",
                    "size": min(28, 12 + 1.2 * degree.get(cid, 1)),
                    "color": color,
                    "shape": shape,
                    "title": title,
                }
            )
            seen.add(cid)
        vedges.append({"from": doc_id, "to": cid, "color": "#D5D5DE"})
        if show_entities:
            for ename in chunk["entities"]:
                if ename not in seen:
                    vnodes.append(_vis_entity(ename, entity_types, degree, labeled(ename)))
                    seen.add(ename)
                vedges.append({"from": cid, "to": ename, "color": "#E2E2EA"})

    if show_entities:
        for rel in relationships:
            for endpoint in (rel["source"], rel["target"]):
                if endpoint not in seen:
                    vnodes.append(
                        _vis_entity(endpoint, entity_types, degree, labeled(endpoint))
                    )
                    seen.add(endpoint)
            vedges.append(
                {
                    "from": rel["source"],
                    "to": rel["target"],
                    "label": rel["relation"] if show_rel_labels else "",
                    "color": "#B79ECC",
                }
            )

    legend_items = [(DOC_COLOR, "Document"), (CHUNK_COLOR, "Chunk")]
    # Show a legend entry for each non-text block type actually present.
    present_blocks = {c.get("block_type", "text") for c in chunks} - {"text"}
    for bt in sorted(present_blocks):
        color, _, label = BLOCK_STYLE.get(bt, BLOCK_STYLE["text"])
        legend_items.append((color, label))
    if show_entities:
        present_types = sorted(
            {(entity_types.get(e) or "CONCEPT").upper() for e in ent_names}
        )
        legend_items += [(_entity_color(t), t.title()) for t in present_types]
    st.markdown(_legend_html(legend_items), unsafe_allow_html=True)
    n_labelled = sum(1 for n in vnodes if n.get("label", "").strip())
    st.caption(
        f"{len(vnodes)} nodes · {len(vedges)} edges · {n_labelled} labelled "
        "· hover a node for its name, drag to rearrange, scroll to zoom"
    )
    components.html(_vis_network_html(vnodes, vedges, int(spacing)), height=660)


def _vis_entity(
    name: str, entity_types: dict[str, str], degree: dict[str, int], labeled: bool
) -> dict[str, Any]:
    """Build a vis.js node dict for an entity (type-coloured, degree-sized)."""
    etype = (entity_types.get(name) or "CONCEPT").upper()
    return {
        "id": name,
        "label": name[:22] if labeled else "",
        "size": min(30, 10 + 2.0 * degree.get(name, 1)),
        "color": _entity_color(etype),
        "shape": "dot",
        "title": f"{name} ({etype}) · degree {degree.get(name, 1)}",
    }


def _vis_network_html(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], spacing: int
) -> str:
    """Return a self-contained HTML page rendering a vis.js network.

    Args:
        nodes: vis.js node dicts.
        edges: vis.js edge dicts.
        spacing: Controls node repulsion/spring length (higher = more spread).

    Returns:
        An HTML string to embed via ``components.html``.
    """
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    gravity = -55 * spacing
    return f"""
<div id="net" style="width:100%;height:625px;background:#fff;
     border:1px solid #e6e6ef;border-radius:10px;"></div>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<script>
  (function() {{
    var container = document.getElementById('net');
    var data = {{
      nodes: new vis.DataSet({nodes_json}),
      edges: new vis.DataSet({edges_json})
    }};
    var options = {{
      nodes: {{ font: {{ size: 14, face: 'sans-serif' }}, borderWidth: 1,
                shadow: {{ enabled: true, size: 4, x: 0, y: 1,
                           color: 'rgba(0,0,0,0.08)' }} }},
      edges: {{ arrows: {{ to: {{ enabled: true, scaleFactor: 0.45 }} }},
                smooth: {{ type: 'continuous' }}, color: {{ inherit: false }},
                width: 1,
                font: {{ size: 10, color: '#6b6b7b', strokeWidth: 4,
                         strokeColor: '#ffffff' }} }},
      physics: {{ solver: 'barnesHut',
                  barnesHut: {{ gravitationalConstant: {gravity},
                               centralGravity: 0.3, springLength: {spacing},
                               springConstant: 0.03, damping: 0.16,
                               avoidOverlap: 0.55 }},
                  stabilization: {{ enabled: true, iterations: 300, fit: true }} }},
      interaction: {{ hover: true, tooltipDelay: 120, dragNodes: true,
                      zoomView: true, navigationButtons: false }}
    }};
    var network = new vis.Network(container, data, options);
    network.once('stabilizationIterationsDone', function() {{
      network.setOptions({{ physics: false }});
      network.fit({{ animation: false }});
    }});
  }})();
</script>
"""


def _render_query_projection(
    res: dict[str, Any],
    chunks: list[dict[str, Any]],
    df: pd.DataFrame,
    mean: np.ndarray,
    comps: np.ndarray,
) -> None:
    """Overlay the query point and retrieval links on the embedding space.

    Args:
        res: The embed-query API response ``{embedding, retrieved}``.
        chunks: The document's chunk dicts (aligned with ``df`` rows).
        df: The chunk 2D-projection dataframe from the embedding section.
        mean: PCA column mean used to project the query consistently.
        comps: PCA (2, d) components used to project the query.
    """
    q_emb = np.asarray(res["embedding"], dtype=float)
    q2d = (q_emb - mean) @ comps.T
    retrieved = {r["chunk_id"]: r["score"] for r in res["retrieved"]}

    df2 = df.copy()
    df2["status"] = [
        "retrieved" if c["chunk_id"] in retrieved else "other" for c in chunks
    ]
    df2["score"] = [
        round(retrieved.get(c["chunk_id"], float("nan")), 4) for c in chunks
    ]

    q_x, q_y = float(q2d[0]), float(q2d[1])
    qdf = pd.DataFrame({"x": [q_x], "y": [q_y], "label": ["QUERY"]})
    line_rows = [
        {"x": q_x, "y": q_y, "x2": float(px), "y2": float(py),
         "score": round(retrieved[c["chunk_id"]], 4)}
        for c, px, py in zip(chunks, df["x"], df["y"])
        if c["chunk_id"] in retrieved
    ]

    pts = (
        alt.Chart(df2)
        .mark_circle(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", title="PC-1"),
            y=alt.Y("y:Q", title="PC-2"),
            size=alt.Size(
                "status:N",
                scale=alt.Scale(
                    domain=["retrieved", "other"], range=[320, 130]
                ),
                legend=None,
            ),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=["retrieved", "other"],
                    range=[RETRIEVED_COLOR, DIMMED_COLOR],
                ),
                title="",
            ),
            tooltip=["chunk", "page", "preview", "score"],
        )
    )
    layers: list[alt.Chart] = []
    if line_rows:
        layers.append(
            alt.Chart(pd.DataFrame(line_rows))
            .mark_rule(color=RETRIEVED_COLOR, opacity=0.5)
            .encode(
                x="x:Q",
                y="y:Q",
                x2="x2:Q",
                y2="y2:Q",
                strokeWidth=alt.StrokeWidth(
                    "score:Q", legend=None, scale=alt.Scale(range=[1, 5])
                ),
                tooltip=["score"],
            )
        )
    layers.append(pts)
    # Halo + marker for the query point.
    layers.append(
        alt.Chart(qdf)
        .mark_point(shape="triangle-up", size=1100, filled=True,
                    color=QUERY_COLOR, opacity=0.18)
        .encode(x="x:Q", y="y:Q")
    )
    layers.append(
        alt.Chart(qdf)
        .mark_point(shape="triangle-up", size=520, filled=True,
                    color=QUERY_COLOR, stroke="white", strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q", tooltip=["label"])
    )
    layers.append(
        alt.Chart(qdf)
        .mark_text(dy=-24, color=QUERY_COLOR, fontWeight="bold", fontSize=13)
        .encode(x="x:Q", y="y:Q", text="label")
    )
    st.markdown(
        _legend_html(
            [(QUERY_COLOR, "Your query"),
             (RETRIEVED_COLOR, "Retrieved (top-k)"),
             (DIMMED_COLOR, "Other chunks")]
        ),
        unsafe_allow_html=True,
    )
    st.altair_chart(
        alt.layer(*layers).interactive().properties(height=470),
        use_container_width=True,
    )

    st.markdown("**Retrieved chunks (nearest vectors, across all documents):**")
    cid_to_chunk = {c["chunk_id"]: c for c in chunks}
    rows = []
    for r in res["retrieved"]:
        chunk = cid_to_chunk.get(r["chunk_id"])
        rows.append(
            {
                "rank": len(rows) + 1,
                "chunk_id": r["chunk_id"],
                "score": round(r["score"], 4),
                "in this doc": r["chunk_id"] in cid_to_chunk,
                "text": (chunk["text"][:130].replace("\n", " ") + "…")
                if chunk
                else "(belongs to another document)",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main() -> None:
    """Application entry point rendering the three tabs."""
    st.title("🕸️ GraphRAG")
    mode, top_k = render_sidebar()
    tab1, tab2, tab3 = st.tabs(
        ["Document Q&A", "Knowledge Graph Explorer", "Pipeline Visualizer"]
    )
    with tab1:
        render_qa_tab(mode, top_k)
    with tab2:
        render_graph_tab()
    with tab3:
        render_pipeline_tab()


if __name__ == "__main__":
    main()
