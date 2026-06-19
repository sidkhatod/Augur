"""
layer1.py

Layer 1 orchestrator: Sensory & Early Fusion.

Runs three async steps in parallel where possible:
  1. Text embedding     (Gemini API call)
  2. Tabular encoding   (FT-Transformer, local inference)
  → Both complete → 3. Fusion (GatedFusion, local)

Outputs a FusedVector ready for:
  - Layer 3 safety guardrails (preliminary_risk_score reflex check)
  - Layer 2 Kenko dialogue engine (fused vector as context)
  - Layer 4 insight engine (stored per session)
"""

import asyncio
import logging
import time

from layer1.embedding_service import EmbeddingService, get_embedding_service, DEFAULT_DIM
from layer1.tabular_encoder import TabularEncoder
from layer1.fusion import FusionLayer
from layer1.models.schemas import Layer1Input, FusedVector, EmbeddingProvider

logger = logging.getLogger(__name__)


class Layer1Pipeline:
    """
    Dependency-injectable orchestrator.
    Accepts pre-built services so tests can inject mocks cleanly.

    Usage (production):
        pipeline = Layer1Pipeline.default()
        result = await pipeline.process(layer1_input)

    Usage (testing):
        pipeline = Layer1Pipeline(
            embedding_service=MockEmbeddingService(),
            tabular_encoder=MockTabularEncoder(),
            fusion_layer=MockFusionLayer(),
        )
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        tabular_encoder: TabularEncoder,
        fusion_layer: FusionLayer,
        embedding_dim: int = DEFAULT_DIM,
    ):
        self.embedding_service = embedding_service
        self.tabular_encoder = tabular_encoder
        self.fusion_layer = fusion_layer
        self.embedding_dim = embedding_dim

    @classmethod
    def default(
        cls,
        provider: EmbeddingProvider = EmbeddingProvider.GEMINI,
        embedding_dim: int = DEFAULT_DIM,
        tabular_model_path: str | None = None,
        fusion_model_path: str | None = None,
        device: str = "cpu",
    ) -> "Layer1Pipeline":
        """
        Convenience constructor for production use.
        Phase 2: change provider= to EmbeddingProvider.DEEPINFRA (one line).
        """
        return cls(
            embedding_service=get_embedding_service(provider),
            tabular_encoder=TabularEncoder(model_path=tabular_model_path, device=device),
            fusion_layer=FusionLayer(model_path=fusion_model_path, device=device),
            embedding_dim=embedding_dim,
        )

    async def process(self, input_data: Layer1Input) -> FusedVector:
        """
        Run the full Layer 1 pipeline for a single user message.

        Text embedding and tabular encoding run concurrently (asyncio.gather).
        Fusion runs after both complete.

        Returns FusedVector with:
          - 832-dim fused vector
          - preliminary_risk_score in [0, 1]
          - full provenance (provider, missing features, latency)
        """
        t0 = time.perf_counter()
        logger.info("Layer1: processing user=%s message_len=%d", input_data.user_id, len(input_data.message))

        # ── Step 1 + 2: run concurrently ──────────────────────────────────────
        text_task = self.embedding_service.embed(
            input_data.message,
            output_dim=self.embedding_dim,
        )
        tabular_task = self.tabular_encoder.encode(input_data.logs)

        text_output, tabular_output = await asyncio.gather(text_task, tabular_task)

        # ── Step 3: fuse ──────────────────────────────────────────────────────
        fused = await self.fusion_layer.fuse(
            user_id=input_data.user_id,
            text_output=text_output,
            tabular_output=tabular_output,
        )

        total_ms = (time.perf_counter() - t0) * 1000
        fused.metadata["total_latency_ms"] = round(total_ms, 1)

        logger.info(
            "Layer1: complete user=%s risk=%.3f latency=%.1fms",
            input_data.user_id, fused.preliminary_risk_score, total_ms,
        )

        # ── Reflex check: log high-risk signals immediately ───────────────────
        # Layer 3 (Llama Guard) does the authoritative check;
        # this is an early-warning log for monitoring dashboards.
        if fused.preliminary_risk_score > 0.7:
            logger.warning(
                "Layer1 REFLEX: high preliminary risk user=%s score=%.3f — forwarding to Layer 3",
                input_data.user_id, fused.preliminary_risk_score,
            )

        return fused
