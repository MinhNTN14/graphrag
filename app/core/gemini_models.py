"""Dynamic Gemini model selection with newest-first fallback.

Instead of hard-coding a single model id (which breaks the moment Google
retires it), the app queries the API's ``list_models`` once per process and
picks the newest *available* model from a priority list, falling back to older
ones. If listing fails (offline / permission), a broadly-available static
default is used so the app still starts.
"""

from __future__ import annotations

from functools import lru_cache

from google import genai
from loguru import logger

from app.config import Settings, get_settings

# Priority order, newest first. The first entry actually available to the API
# key (and supporting the required method) wins. "*-latest" aliases track the
# newest stable model, so they sit at the top as safe "newest" choices.
GENERATION_PRIORITY: list[str] = [
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-pro-latest",
    "gemini-2.5-pro",
]
EMBEDDING_PRIORITY: list[str] = [
    "gemini-embedding-001",
    "gemini-embedding-2",
    "text-embedding-004",
]

# Used only if list_models cannot be reached; must be broadly available.
_GENERATION_FALLBACK = "gemini-2.5-flash"
_EMBEDDING_FALLBACK = "gemini-embedding-001"

# Process-level caches for the resolved bare model names.
_generation_model: str | None = None
_embedding_model: str | None = None


@lru_cache(maxsize=2)
def _client_for(api_key: str) -> genai.Client:
    """Return a cached ``genai.Client`` for ``api_key``."""
    return genai.Client(api_key=api_key)


def get_client(settings: Settings | None = None) -> genai.Client:
    """Return the shared Gemini client for the configured API key.

    The client is cached per API key so every collaborator (embedder,
    extractor, generator) reuses one instance instead of opening its own.

    Args:
        settings: Optional settings override.

    Returns:
        A configured :class:`google.genai.Client`.
    """
    settings = settings or get_settings()
    return _client_for(settings.gemini_api_key)


@lru_cache(maxsize=4)
def _available(method: str) -> frozenset[str]:
    """Return the bare names of models supporting ``method``.

    Args:
        method: A supported action, e.g. ``"generateContent"`` or
            ``"embedContent"``.

    Returns:
        A frozenset of bare model names (no ``models/`` prefix), or an empty
        set if the API could not be listed.
    """
    try:
        names: set[str] = set()
        for model in get_client().models.list():
            # The google-genai SDK exposes the capability list as
            # ``supported_actions`` (the legacy SDK called it
            # ``supported_generation_methods``).
            if method in (getattr(model, "supported_actions", None) or []):
                names.add((model.name or "").split("/")[-1])
        return frozenset(names)
    except Exception as exc:  # noqa: BLE001 - listing is best-effort
        logger.warning(f"Could not list Gemini models ({exc}); using static default")
        return frozenset()


def _select(
    priority: list[str], method: str, fallback: str, kind: str
) -> str:
    """Pick the newest available model for ``method`` from ``priority``.

    Args:
        priority: Candidate model names, newest first.
        method: The required generation method.
        fallback: Static default if the API cannot be listed.
        kind: Human label for logging ("generation"/"embedding").

    Returns:
        The chosen bare model name.
    """
    available = _available(method)
    if available:
        for candidate in priority:
            if candidate in available:
                logger.info(f"Selected {kind} model: {candidate}")
                return candidate
        chosen = sorted(available)[0]
        logger.warning(
            f"No preferred {kind} model available; using {chosen}"
        )
        return chosen
    logger.info(f"Using static {kind} model: {fallback}")
    return fallback


def resolve_generation_model(settings: Settings | None = None) -> str:
    """Return the resolved bare generation model name (cached).

    Args:
        settings: Optional settings override.

    Returns:
        A bare model name, e.g. ``"gemini-flash-latest"``.
    """
    global _generation_model
    if _generation_model is None:
        settings = settings or get_settings()
        _generation_model = _select(
            GENERATION_PRIORITY, "generateContent", _GENERATION_FALLBACK, "generation"
        )
    return _generation_model


def resolve_embedding_model(settings: Settings | None = None) -> str:
    """Return the resolved embedding model name with ``models/`` prefix (cached).

    Args:
        settings: Optional settings override.

    Returns:
        A prefixed model name, e.g. ``"models/gemini-embedding-001"``.
    """
    global _embedding_model
    if _embedding_model is None:
        settings = settings or get_settings()
        name = _select(
            EMBEDDING_PRIORITY, "embedContent", _EMBEDDING_FALLBACK, "embedding"
        )
        _embedding_model = f"models/{name}"
    return _embedding_model
