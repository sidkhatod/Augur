"""
tabular_encoder.py

FT-Transformer (Feature Tokenisation Transformer) for encoding structured
user self-report logs into a dense vector.

Architecture (Gorishniy et al., 2021 — "Revisiting Deep Learning Models for Tabular Data"):
  raw features → per-feature linear tokenisation → transformer (self-attention across features)
  → CLS token output → linear projection → 64-dim tabular embedding

Why FT-Transformer over plain MLP:
  - Learns cross-feature interactions explicitly via self-attention
    (e.g. low sleep + low social = qualitatively different risk signal)
  - Handles missing features gracefully via learned mask tokens
  - Output is a differentiable embedding ready for fusion with text

Phase 2 upgrade path:
  - Add temporal encoder (PatchTST or iTransformer) over a 7-day log window
  - Gate-fuse temporal vector with today's FT-Transformer snapshot vector
  - See architecture doc §tabular_encoder for the dual-path design
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from layer1.models.schemas import UserLogs, TabularEmbeddingOutput

logger = logging.getLogger(__name__)


# ── Feature config ────────────────────────────────────────────────────────────

# Ordered feature list — order must stay stable across training and inference.
# To add a feature: append to this list, retrain, bump MODEL_VERSION.
FEATURE_NAMES: list[str] = [
    "sleep_hours",       # 0–24
    "mood_score",        # 1–10
    "productivity_score",# 1–10
    "social_score",      # 1–10
    "session_frequency", # 0–∞ (clipped at 30 for normalisation)
]

# Min-max bounds for normalisation → [0, 1]
# Derived from StudentLife/CES dataset distributions + clinical reasoning
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "sleep_hours":        (0.0,  12.0),
    "mood_score":         (1.0,  10.0),
    "productivity_score": (1.0,  10.0),
    "social_score":       (1.0,  10.0),
    "session_frequency":  (0.0,  30.0),
}

# Global mean for each feature — used as imputation value when log is missing.
# These are rough priors; replace with dataset-computed means after training.
FEATURE_MEANS: dict[str, float] = {
    "sleep_hours":        7.0,
    "mood_score":         6.0,
    "productivity_score": 6.0,
    "social_score":       5.0,
    "session_frequency":  3.0,
}

N_FEATURES = len(FEATURE_NAMES)
TABULAR_EMBED_DIM = 64     # Output dimensionality — matches fusion layer expectation
MODEL_VERSION = "ft-transformer-v1.0"


# ── FT-Transformer components ─────────────────────────────────────────────────

class FeatureTokeniser(nn.Module):
    """
    Projects each scalar feature into a D-dimensional token independently.
    Each feature gets its own learned weight + bias (no shared projection).

    For feature i with value x_i:
        token_i = x_i * W_i + b_i    (W_i ∈ ℝ^D, b_i ∈ ℝ^D)

    Missing features are replaced with a learned mask token before this step.
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        # Per-feature projection: n_features separate linear projections
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        # Learned mask token — substituted when a feature is missing
        self.mask_token = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(
        self,
        x: torch.Tensor,           # (batch, n_features)  normalised values
        missing_mask: torch.Tensor  # (batch, n_features)  True = missing
    ) -> torch.Tensor:              # (batch, n_features, d_token)
        # (batch, n_features, 1) * (n_features, d_token) → (batch, n_features, d_token)
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        # Replace missing feature tokens with the learned mask token
        mask = missing_mask.unsqueeze(-1).expand_as(tokens)
        tokens = torch.where(mask, self.mask_token.unsqueeze(0).expand_as(tokens), tokens)
        return tokens


class FTTransformerEncoder(nn.Module):
    """
    Full FT-Transformer encoder.

    Pipeline:
      1. Feature tokenisation  (each scalar → D-dim token)
      2. Prepend [CLS] token
      3. Transformer (self-attention across ALL n_features + 1 tokens)
      4. Extract [CLS] token output → summary of all feature interactions
      5. Linear projection → tabular_embed_dim

    Hyperparameters are deliberately small — 5 features is a tiny input space.
    Scale up d_token and n_layers when feature count grows past ~20.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        tabular_embed_dim: int = TABULAR_EMBED_DIM,
    ):
        super().__init__()
        self.feature_tokeniser = FeatureTokeniser(n_features, d_token)

        # Learned CLS token (prepended before transformer input)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm: more stable for small models
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Project CLS output → final tabular embedding
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, tabular_embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,            # (batch, n_features)
        missing_mask: torch.Tensor  # (batch, n_features)
    ) -> torch.Tensor:              # (batch, tabular_embed_dim)
        batch_size = x.size(0)

        # Tokenise features: (batch, n_features, d_token)
        tokens = self.feature_tokeniser(x, missing_mask)

        # Prepend CLS token: (batch, n_features + 1, d_token)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Self-attention across all feature tokens + CLS
        out = self.transformer(tokens)

        # Extract CLS (position 0) and project
        cls_out = out[:, 0, :]
        return self.output_projection(cls_out)


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def preprocess_logs(logs: UserLogs) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Convert UserLogs Pydantic model → normalised tensor + missing mask.

    Returns:
        x            (1, n_features)  float32, normalised to [0, 1]
        missing_mask (1, n_features)  bool, True where feature was None
        missing_names                 list of feature names that were imputed
    """
    values = []
    missing_mask = []
    missing_names = []
    log_dict = logs.model_dump()

    for name in FEATURE_NAMES:
        raw = log_dict.get(name)
        if raw is None:
            # Impute with global mean, flag as missing
            imputed = FEATURE_MEANS[name]
            values.append(_normalise(imputed, name))
            missing_mask.append(True)
            missing_names.append(name)
        else:
            values.append(_normalise(float(raw), name))
            missing_mask.append(False)

    x = torch.tensor([values], dtype=torch.float32)
    mask = torch.tensor([missing_mask], dtype=torch.bool)
    return x, mask, missing_names


def _normalise(value: float, feature_name: str) -> float:
    """Min-max normalise a single feature value to [0, 1], clipping to bounds."""
    lo, hi = FEATURE_BOUNDS[feature_name]
    clipped = max(lo, min(hi, value))
    return (clipped - lo) / (hi - lo) if hi > lo else 0.0


# ── Public interface ──────────────────────────────────────────────────────────

class TabularEncoder:
    """
    Stateless wrapper around FTTransformerEncoder for use in Layer 1.

    Usage:
        encoder = TabularEncoder()
        output = await encoder.encode(logs)
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = FTTransformerEncoder().to(self.device)
        self.model.eval()

        if model_path:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            logger.info("TabularEncoder: loaded weights from %s", model_path)
        else:
            logger.warning(
                "TabularEncoder: no weights loaded — running with random initialisation. "
                "This is expected during development; load trained weights before production."
            )

    async def encode(self, logs: UserLogs) -> TabularEmbeddingOutput:
        """
        Encode a single UserLogs snapshot into a 64-dim tabular embedding.
        Handles missing features transparently via learned mask tokens.
        """
        x, missing_mask, missing_names = preprocess_logs(logs)

        with torch.no_grad():
            x = x.to(self.device)
            missing_mask = missing_mask.to(self.device)
            embedding = self.model(x, missing_mask)

        vector = embedding.squeeze(0).cpu().tolist()

        if missing_names:
            logger.debug("TabularEncoder: imputed missing features: %s", missing_names)

        return TabularEmbeddingOutput(
            vector=vector,
            dim=TABULAR_EMBED_DIM,
            missing_features=missing_names,
        )

    def get_model(self) -> FTTransformerEncoder:
        """Expose underlying model for training/fine-tuning scripts."""
        return self.model
