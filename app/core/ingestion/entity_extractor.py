"""Entity and relationship extraction using Gemini.

Prompts ``gemini-1.5-flash`` to return a strict JSON object describing entities
and relationships found in a piece of text. Parsing failures are handled
gracefully by returning an empty structure rather than raising.
"""

from __future__ import annotations

import json
import re
from typing import Any

import google.generativeai as genai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from app.core.gemini_models import resolve_generation_model

_PROMPT_TEMPLATE = """Extract all entities and relationships from the text below.

Return ONLY a valid JSON object with this exact structure, no markdown, no explanation:
{{
  "entities": [
    {{"name": "string", "type": "PERSON|ORGANIZATION|CONCEPT|TECHNOLOGY|LOCATION|EVENT", "description": "one sentence"}}
  ],
  "relationships": [
    {{"source": "entity_name", "relation": "VERB_IN_CAPS", "target": "entity_name", "description": "one sentence"}}
  ]
}}

Rules:
- Normalize entity names (e.g. "Apache Kafka" not "kafka")
- Only extract entities explicitly mentioned, never infer
- Relation types: active voice verbs in UPPER_SNAKE_CASE (USES, DEVELOPED_BY, WORKS_AT, PART_OF, IMPLEMENTS, TRAINED_ON)
- Both source and target must appear in the entities list
- If no entities found, return {{"entities": [], "relationships": []}}

Text:
{text}
"""


class EntityExtractor:
    """Extract entities and relationships from text via Gemini."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Configure the Gemini generative model.

        Args:
            settings: Optional settings override.
        """
        self._settings = settings or get_settings()
        genai.configure(api_key=self._settings.gemini_api_key)
        self._model = genai.GenerativeModel(resolve_generation_model(self._settings))

    def extract(self, text: str) -> dict[str, Any]:
        """Extract entities and relationships from ``text``.

        Never raises on parse failure: malformed model output yields an empty
        ``{"entities": [], "relationships": []}`` structure.

        Args:
            text: The source text to analyse.

        Returns:
            A dict with ``entities`` and ``relationships`` lists.
        """
        empty: dict[str, Any] = {"entities": [], "relationships": []}
        if not text or not text.strip():
            return empty
        try:
            raw = self._call_model(text)
        except Exception as exc:  # noqa: BLE001 - retries already exhausted
            logger.warning(f"Entity extraction call failed: {exc}")
            return empty

        parsed = self._parse_response(raw)
        if parsed is None:
            logger.warning("Failed to parse entity extraction JSON; returning empty")
            return empty
        return parsed

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=4, max=64))
    def _call_model(self, text: str) -> str:
        """Call Gemini and return the raw text response.

        Args:
            text: The source text.

        Returns:
            The raw model response text.
        """
        prompt = _PROMPT_TEMPLATE.format(text=text)
        logger.debug(f"Extracting entities from text of length {len(text)}")
        response = self._model.generate_content(prompt)
        return response.text or ""

    def _parse_response(self, raw: str) -> dict[str, Any] | None:
        """Parse the model response into the expected structure.

        Strips markdown code fences if present. Returns ``None`` on failure.

        Args:
            raw: Raw model output.

        Returns:
            The parsed dict or ``None``.
        """
        cleaned = raw.strip()
        # Remove ```json ... ``` fences if the model added them.
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        # Fall back to extracting the first {...} block.
        if not cleaned.startswith("{"):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        if not isinstance(entities, list) or not isinstance(relationships, list):
            return None
        return {"entities": entities, "relationships": relationships}
