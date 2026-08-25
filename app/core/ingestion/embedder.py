"""Gemini text embedding wrapper.

Wraps ``google.generativeai.embed_content`` with tenacity retries. Uses the
``models/text-embedding-004`` model which produces 768-dimensional vectors.
"""

from __future__ import annotations

import google.generativeai as genai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from app.core.gemini_models import resolve_embedding_model

# The vector index is created with 768 dimensions; request that size
# explicitly (the model would otherwise default to a larger dimensionality).
_EMBEDDING_DIMENSIONS = 768


class GeminiEmbedder:
    """Produce embeddings for text using the Gemini embedding API."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Configure the Gemini client from application settings.

        Args:
            settings: Optional settings override.
        """
        self._settings = settings or get_settings()
        genai.configure(api_key=self._settings.gemini_api_key)
        self._model = resolve_embedding_model(self._settings)

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=4, max=64))
    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A 768-dimensional embedding vector.
        """
        logger.debug(f"Embedding text of length {len(text)} with {self._model}")
        kwargs: dict[str, object] = {}
        # Only the gemini-embedding-* models accept a custom output dimension;
        # text-embedding-004 already returns 768 dims natively.
        if "gemini-embedding" in self._model:
            kwargs["output_dimensionality"] = _EMBEDDING_DIMENSIONS
        response = genai.embed_content(model=self._model, content=text, **kwargs)
        # The SDK returns either a dict with an 'embedding' key or an object.
        if isinstance(response, dict):
            return list(response["embedding"])
        return list(response.embedding)  # type: ignore[attr-defined]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts one by one.

        The Gemini free tier does not expose a batch endpoint, so this simply
        loops over ``embed``.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors in the same order as the input.
        """
        return [self.embed(text) for text in texts]
