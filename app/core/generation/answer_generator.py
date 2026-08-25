"""Answer generation from retrieved context using Gemini."""

from __future__ import annotations

from typing import Any

import google.generativeai as genai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from app.core.gemini_models import resolve_generation_model

_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided context.

Context from documents:
{context}

{graph_section}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the context does not contain enough information, say so clearly
- Be concise and factual
- Cite which document/chunk your answer comes from

Answer:"""


class AnswerGenerator:
    """Generate grounded answers from retrieved chunks and graph context."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Configure the Gemini generative model.

        Args:
            settings: Optional settings override.
        """
        self._settings = settings or get_settings()
        genai.configure(api_key=self._settings.gemini_api_key)
        self._model = genai.GenerativeModel(resolve_generation_model(self._settings))

    def generate(self, question: str, retrieval_result: dict[str, Any]) -> dict[str, Any]:
        """Generate an answer for ``question`` from a retrieval result.

        Args:
            question: The user's question.
            retrieval_result: Output of :class:`HybridRetriever.retrieve`.

        Returns:
            A dict with ``answer``, ``sources``, ``graph_context``,
            ``mode_used`` and ``entities_found``.
        """
        chunks = retrieval_result.get("chunks", [])
        graph_context = retrieval_result.get("graph_context", [])
        mode_used = retrieval_result.get("mode_used", "hybrid")
        entities_found = retrieval_result.get("entities_found", [])

        context = self._build_context(chunks)
        graph_section = self._build_graph_section(graph_context)

        if not chunks and not graph_context:
            answer = (
                "The provided context does not contain enough information to "
                "answer this question."
            )
        else:
            answer = self._call_model(question, context, graph_section)

        sources = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", ""),
                "doc_id": c.get("doc_id", ""),
                "score": c.get("score"),
                "source": c.get("source", mode_used),
            }
            for c in chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "graph_context": graph_context,
            "mode_used": mode_used,
            "entities_found": entities_found,
        }

    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        """Build a numbered context string from chunks.

        Args:
            chunks: Retrieved chunk dicts.

        Returns:
            A formatted context string, or a placeholder if empty.
        """
        if not chunks:
            return "(no document chunks retrieved)"
        parts: list[str] = []
        for chunk in chunks:
            header = (
                f"[doc={chunk.get('doc_id', '?')} "
                f"chunk={chunk.get('chunk_id', '?')}]"
            )
            parts.append(f"{header}\n{chunk.get('text', '')}")
        return "\n\n".join(parts)

    def _build_graph_section(self, graph_context: list[dict[str, Any]]) -> str:
        """Build the relationship-summary section of the prompt.

        Args:
            graph_context: Relationship path dicts.

        Returns:
            A formatted graph section, empty string if no paths.
        """
        if not graph_context:
            return ""
        lines = ["Relevant entity relationships from the knowledge graph:"]
        for path in graph_context:
            lines.append(
                f"- {path.get('source', '?')} "
                f"-[{path.get('relation', 'RELATED_TO')}]-> "
                f"{path.get('target', '?')}"
                + (f" ({path['description']})" if path.get("description") else "")
            )
        return "\n".join(lines)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_model(self, question: str, context: str, graph_section: str) -> str:
        """Call Gemini to produce the final answer text.

        Args:
            question: The user's question.
            context: The document context block.
            graph_section: The graph relationship block.

        Returns:
            The generated answer text.
        """
        prompt = _PROMPT_TEMPLATE.format(
            context=context, graph_section=graph_section, question=question
        )
        logger.debug("Generating answer from context")
        response = self._model.generate_content(prompt)
        return (response.text or "").strip()
