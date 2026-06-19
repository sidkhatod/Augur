"""
embedding_service.py

Abstract EmbeddingService interface + Phase 1 Gemini implementation.

Phase 2 migration path:
  - Implement DeepInfraEmbeddingService(EmbeddingService) pointing at Qwen3-Embedding-4B
  - Swap one line in layer1.py: embedding_service = DeepInfraEmbeddingService(...)
  - Re-index vector DB — nothing else changes.
"""

import os
import time
import logging
from abc import ABC, abstractmethod

from google import genai
from google.genai import types as genai_types
import numpy as np

from layer1.models.schemas import TextEmbeddingOutput, EmbeddingProvider

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-embedding-2"

# Matryoshka: choose output dim based on downstream task.
# Risk classifier  → 768 (full precision)
# RAG vector DB    → 256 (cheaper retrieval, minimal quality loss)
SUPPORTED_DIMS = {768, 512, 256}
DEFAULT_DIM = 768


# ── Abstract interface ────────────────────────────────────────────────────────

class EmbeddingService(ABC):
    """
    Stable interface for all embedding providers.
    Downstream code (fusion, risk classifier, RAG) only touches this contract.
    """

    @abstractmethod
    async def embed(self, text: str, output_dim: int = DEFAULT_DIM) -> TextEmbeddingOutput:
        """
        Embed a single text string.

        Args:
            text:       Raw input string (chat message, document chunk, etc.)
            output_dim: Desired output dimensionality. Provider must support it.

        Returns:
            TextEmbeddingOutput with .vector, .dim, .provider
        """
        ...

    @abstractmethod
    async def embed_batch(
        self, texts: list[str], output_dim: int = DEFAULT_DIM
    ) -> list[TextEmbeddingOutput]:
        """
        Batch embed for bulk processing (Layer 4 transcript analysis, RAG indexing).
        Implementations should use the provider's native batching to minimise API calls.
        """
        ...


# ── Phase 1: Gemini text-embedding-004 ───────────────────────────────────────

class GeminiEmbeddingService(EmbeddingService):
    """
    Phase 1 implementation using Google Gemini text-embedding-004.

    Matryoshka support: pass output_dim=256/512/768.
    The API truncates internally — no post-processing needed.

    Data privacy note:
        Before production, verify your GCP project is on a HIPAA-eligible
        service agreement and that zero-data-retention is configured.
        See: https://cloud.google.com/vertex-ai/docs/generative-ai/data-governance
    """

    # Task types supported by Gemini text-embedding-004
    # SEMANTIC_SIMILARITY is correct for risk classification and RAG retrieval
    TASK_TYPE = "SEMANTIC_SIMILARITY"

    def __init__(self, api_key: str | None = None, output_dim: int = DEFAULT_DIM):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set. Export it: export GEMINI_API_KEY=your_key")
        if output_dim not in SUPPORTED_DIMS:
            raise ValueError(f"output_dim must be one of {SUPPORTED_DIMS}, got {output_dim}")

        self._client = genai.Client(api_key=api_key)
        self._default_dim = output_dim
        logger.info("GeminiEmbeddingService initialised — model=%s dim=%d", GEMINI_MODEL, output_dim)

    async def embed(self, text: str, output_dim: int | None = None) -> TextEmbeddingOutput:
        dim = output_dim or self._default_dim
        t0 = time.perf_counter()

        result = self._client.models.embed_content(
            model=GEMINI_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(
                task_type=self.TASK_TYPE,
                output_dimensionality=dim,
            ),
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Gemini embed latency=%.1fms dim=%d", latency_ms, dim)

        return TextEmbeddingOutput(
            vector=result.embeddings[0].values,
            dim=dim,
            provider=EmbeddingProvider.GEMINI,
        )

    async def embed_batch(self, texts: list[str], output_dim: int | None = None) -> list[TextEmbeddingOutput]:
        dim = output_dim or self._default_dim
        t0 = time.perf_counter()

        result = self._client.models.embed_content(
            model=GEMINI_MODEL,
            contents=texts,
            config=genai_types.EmbedContentConfig(
                task_type=self.TASK_TYPE,
                output_dimensionality=dim,
            ),
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Gemini batch embed n=%d latency=%.1fms dim=%d", len(texts), latency_ms, dim)

        return [
            TextEmbeddingOutput(vector=emb.values, dim=dim, provider=EmbeddingProvider.GEMINI)
            for emb in result.embeddings
        ]


# ── Convenience factory ───────────────────────────────────────────────────────

def get_embedding_service(provider: EmbeddingProvider = EmbeddingProvider.GEMINI) -> EmbeddingService:
    """
    Factory function — keeps provider instantiation in one place.

    Phase 2: add elif provider == EmbeddingProvider.DEEPINFRA: return DeepInfraEmbeddingService(...)
    """
    if provider == EmbeddingProvider.GEMINI:
        return GeminiEmbeddingService()
    raise NotImplementedError(f"Provider {provider} not yet implemented. See Phase 2 migration guide.")
