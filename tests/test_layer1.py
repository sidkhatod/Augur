"""
tests/test_layer1.py

Unit tests for Layer 1 components.
Tabular encoder and fusion layer are tested without API keys (no Gemini calls).
A mock embedding service is used to test the full pipeline integration.
"""

import asyncio
import pytest
import torch

from layer1.models.schemas import UserLogs, Layer1Input, EmbeddingProvider, TextEmbeddingOutput
from layer1.tabular_encoder import (
    TabularEncoder,
    preprocess_logs,
    FEATURE_NAMES,
    TABULAR_EMBED_DIM,
)
from layer1.fusion import FusionLayer, FUSED_DIM, TEXT_DIM, TABULAR_DIM
from layer1.layer1 import Layer1Pipeline
from layer1.embedding_service import EmbeddingService


# ── Helpers ───────────────────────────────────────────────────────────────────

class MockEmbeddingService(EmbeddingService):
    """Returns a deterministic zero vector — no API call."""
    async def embed(self, text, output_dim=768):
        return TextEmbeddingOutput(
            vector=[0.0] * output_dim,
            dim=output_dim,
            provider=EmbeddingProvider.GEMINI,
        )
    async def embed_batch(self, texts, output_dim=768):
        return [await self.embed(t, output_dim) for t in texts]


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Preprocessing tests ───────────────────────────────────────────────────────

def test_preprocess_all_present():
    logs = UserLogs(sleep_hours=7.0, mood_score=6.0, productivity_score=7.0, social_score=5.0, session_frequency=3.0)
    x, mask, missing = preprocess_logs(logs)
    assert x.shape == (1, len(FEATURE_NAMES))
    assert not mask.any(), "No features should be masked"
    assert missing == []
    # Sleep 7h out of [0,12] → 7/12 ≈ 0.583
    assert abs(x[0, 0].item() - 7/12) < 1e-4


def test_preprocess_missing_features():
    logs = UserLogs(sleep_hours=None, mood_score=5.0)
    x, mask, missing = preprocess_logs(logs)
    assert "sleep_hours" in missing
    assert "productivity_score" in missing
    assert "social_score" in missing
    assert "session_frequency" in missing
    assert mask[0, 0].item() is True  # sleep is missing
    assert mask[0, 1].item() is False  # mood is present


def test_preprocess_normalisation_bounds():
    """Values at bounds should normalise to 0.0 and 1.0."""
    logs = UserLogs(sleep_hours=0.0, mood_score=1.0, productivity_score=10.0, social_score=10.0)
    x, _, _ = preprocess_logs(logs)
    assert abs(x[0, 0].item() - 0.0) < 1e-4   # sleep=0 → 0.0
    assert abs(x[0, 1].item() - 0.0) < 1e-4   # mood=1 → 0.0
    assert abs(x[0, 2].item() - 1.0) < 1e-4   # productivity=10 → 1.0


def test_preprocess_out_of_bound_clipped():
    """Values above the normalisation ceiling (12h) but within schema (≤24) should clip to 1.0."""
    logs = UserLogs(sleep_hours=20.0)  # valid for schema (≤24) but above norm ceiling (12)
    x, _, _ = preprocess_logs(logs)
    assert abs(x[0, 0].item() - 1.0) < 1e-4  # clipped to 12 → normalised to 1.0


# ── Tabular encoder tests ─────────────────────────────────────────────────────

def test_tabular_encoder_output_shape():
    encoder = TabularEncoder()
    logs = UserLogs(sleep_hours=6.0, mood_score=5.0, productivity_score=6.0, social_score=4.0)
    output = run(encoder.encode(logs))
    assert output.dim == TABULAR_EMBED_DIM
    assert len(output.vector) == TABULAR_EMBED_DIM


def test_tabular_encoder_missing_logs():
    encoder = TabularEncoder()
    logs = UserLogs()  # All None
    output = run(encoder.encode(logs))
    assert output.dim == TABULAR_EMBED_DIM
    assert set(output.missing_features) == set(FEATURE_NAMES)


def test_tabular_encoder_deterministic():
    """Same input should produce same output (model in eval mode, no dropout)."""
    encoder = TabularEncoder()
    logs = UserLogs(sleep_hours=6.0, mood_score=7.0)
    out1 = run(encoder.encode(logs))
    out2 = run(encoder.encode(logs))
    assert out1.vector == out2.vector


# ── Fusion layer tests ────────────────────────────────────────────────────────

def test_fusion_output_shape():
    fusion = FusionLayer()
    text_out = TextEmbeddingOutput(vector=[0.1]*TEXT_DIM, dim=TEXT_DIM, provider=EmbeddingProvider.GEMINI)
    encoder = TabularEncoder()
    tabular_out = run(encoder.encode(UserLogs(sleep_hours=7.0)))
    result = run(fusion.fuse("user_001", text_out, tabular_out))
    assert result.dim == FUSED_DIM
    assert len(result.vector) == FUSED_DIM


def test_fusion_risk_score_bounds():
    fusion = FusionLayer()
    text_out = TextEmbeddingOutput(vector=[0.0]*TEXT_DIM, dim=TEXT_DIM, provider=EmbeddingProvider.GEMINI)
    encoder = TabularEncoder()
    tabular_out = run(encoder.encode(UserLogs()))
    result = run(fusion.fuse("user_002", text_out, tabular_out))
    assert 0.0 <= result.preliminary_risk_score <= 1.0


# ── Full pipeline integration test ────────────────────────────────────────────

def test_pipeline_end_to_end():
    pipeline = Layer1Pipeline(
        embedding_service=MockEmbeddingService(),
        tabular_encoder=TabularEncoder(),
        fusion_layer=FusionLayer(),
    )
    input_data = Layer1Input(
        user_id="test_user",
        message="I haven't slept in days and feel completely isolated",
        logs=UserLogs(sleep_hours=2.0, mood_score=2.0, social_score=1.0),
    )
    result = run(pipeline.process(input_data))
    assert result.user_id == "test_user"
    assert result.dim == FUSED_DIM
    assert 0.0 <= result.preliminary_risk_score <= 1.0
    assert "total_latency_ms" in result.metadata


def test_pipeline_empty_logs():
    """Pipeline should not crash when user has no logs at all."""
    pipeline = Layer1Pipeline(
        embedding_service=MockEmbeddingService(),
        tabular_encoder=TabularEncoder(),
        fusion_layer=FusionLayer(),
    )
    input_data = Layer1Input(
        user_id="new_user",
        message="Hello",
        logs=UserLogs(),
    )
    result = run(pipeline.process(input_data))
    assert result.dim == FUSED_DIM
    assert len(result.tabular_embedding.missing_features) == len(FEATURE_NAMES)
