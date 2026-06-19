"""
fusion.py

Early fusion: concatenates text embedding (768-dim) and tabular embedding (64-dim)
into a single fused vector (832-dim), then applies a learned gated projection.

Also computes a lightweight preliminary risk score from the fused vector —
this is the "reflex check" that can trigger Layer 3 safety guardrails
before the full LLM dialogue pipeline processes the message.

Architecture:
    text_vec (768) ──┐
                     ├── concat (832) → LayerNorm → Linear → Tanh gate → fused (832)
    tabular_vec (64) ┘                                            ↓
                                                      linear → sigmoid → risk_score (scalar)
"""

import logging

import torch
import torch.nn as nn

from layer1.models.schemas import TextEmbeddingOutput, TabularEmbeddingOutput, FusedVector

logger = logging.getLogger(__name__)

TEXT_DIM = 768
TABULAR_DIM = 64
FUSED_DIM = TEXT_DIM + TABULAR_DIM  # 832


class GatedFusion(nn.Module):
    """
    Learned gated fusion module.

    The gate learns to weight the contribution of text vs tabular signals
    depending on context:
    - If a user just installed the app (no logs), tabular path carries no signal
      → gate suppresses tabular contribution automatically
    - If a user has consistent logs, tabular path strongly informs the fused vector

    Architecture:
        concat → LayerNorm → Linear(FUSED_DIM, FUSED_DIM) → Tanh  [gate]
        gate * concat                                               [gated vector]
        gated_vector → Linear(FUSED_DIM, 1) → Sigmoid             [risk score]
    """

    def __init__(self, fused_dim: int = FUSED_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(fused_dim)

        # Gate: learns how much of each dimension to pass through
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.Tanh(),
        )

        # Preliminary risk classifier head
        # Outputs a scalar [0, 1] — coarse signal before Llama Guard
        self.risk_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        text_vec: torch.Tensor,     # (batch, TEXT_DIM)
        tabular_vec: torch.Tensor,  # (batch, TABULAR_DIM)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_vector  (batch, FUSED_DIM)  — input to risk classifier + Layer 2
            risk_score    (batch, 1)           — preliminary risk [0, 1]
        """
        concat = torch.cat([text_vec, tabular_vec], dim=-1)  # (batch, FUSED_DIM)
        normed = self.norm(concat)
        gate = self.gate(normed)
        fused = gate * concat                                  # element-wise gating
        risk_score = self.risk_head(fused)
        return fused, risk_score


class FusionLayer:
    """
    Stateless wrapper around GatedFusion for use in Layer 1 orchestrator.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = GatedFusion().to(self.device)
        self.model.eval()

        if model_path:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            logger.info("FusionLayer: loaded weights from %s", model_path)
        else:
            logger.warning(
                "FusionLayer: running with random initialisation. "
                "Load trained weights before production."
            )

    async def fuse(
        self,
        user_id: str,
        text_output: TextEmbeddingOutput,
        tabular_output: TabularEmbeddingOutput,
    ) -> FusedVector:
        """
        Fuse text and tabular embeddings into a single FusedVector.
        """
        text_vec = torch.tensor([text_output.vector], dtype=torch.float32).to(self.device)
        tabular_vec = torch.tensor([tabular_output.vector], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            fused, risk_score = self.model(text_vec, tabular_vec)

        fused_list = fused.squeeze(0).cpu().tolist()
        risk_float = float(risk_score.squeeze().cpu().item())

        logger.debug(
            "FusionLayer: user=%s risk_score=%.4f fused_dim=%d",
            user_id, risk_float, len(fused_list),
        )

        return FusedVector(
            user_id=user_id,
            vector=fused_list,
            dim=len(fused_list),
            preliminary_risk_score=risk_float,
            text_embedding=text_output,
            tabular_embedding=tabular_output,
            metadata={
                "text_provider": text_output.provider.value,
                "tabular_missing": tabular_output.missing_features,
                "fused_dim": FUSED_DIM,
            },
        )
