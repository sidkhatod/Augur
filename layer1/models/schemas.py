from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class EmbeddingProvider(str, Enum):
    GEMINI = "AIzaSyD17xyXf8VgjMDpK4ZHtT5jMiOgaeoklBs"
    # Phase 2: DEEPINFRA = "deepinfra" | CLOUD_RUN = "cloud_run"


class UserLogs(BaseModel):
    """
    Structured self-report logs from the app dashboard.
    All fields optional — FT-Transformer handles missing values gracefully.
    Values are raw (pre-normalisation); normalisation happens inside TabularEncoder.
    """
    sleep_hours: Optional[float] = Field(None, ge=0, le=24, description="Hours slept last night")
    mood_score: Optional[float] = Field(None, ge=1, le=10, description="Self-reported mood 1–10")
    productivity_score: Optional[float] = Field(None, ge=1, le=10, description="Self-reported productivity 1–10")
    social_score: Optional[float] = Field(None, ge=1, le=10, description="Self-reported social engagement 1–10")
    session_frequency: Optional[float] = Field(None, ge=0, description="App sessions in last 7 days")
    # Extend here as you add more dashboard metrics

    @field_validator("sleep_hours", "mood_score", "productivity_score", "social_score", "session_frequency", mode="before")
    @classmethod
    def coerce_none_strings(cls, v):
        """Treat empty strings and 'null' strings as None."""
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none"):
            return None
        return v


class Layer1Input(BaseModel):
    """Single input to the Layer 1 pipeline."""
    user_id: str = Field(..., description="Anonymised user identifier")
    message: str = Field(..., min_length=1, description="Raw chat message from user")
    logs: UserLogs = Field(default_factory=UserLogs, description="User self-report logs")


class TextEmbeddingOutput(BaseModel):
    """Output from the text embedding path."""
    vector: list[float] = Field(..., description="Dense text embedding vector")
    dim: int = Field(..., description="Embedding dimensionality")
    provider: EmbeddingProvider


class TabularEmbeddingOutput(BaseModel):
    """Output from the FT-Transformer tabular encoder."""
    vector: list[float] = Field(..., description="Dense tabular embedding vector")
    dim: int = Field(..., description="Embedding dimensionality")
    missing_features: list[str] = Field(default_factory=list, description="Features that were imputed")


class FusedVector(BaseModel):
    """
    Final output of Layer 1: fused representation ready for
    the risk classifier and Layer 2 dialogue engine.
    """
    user_id: str
    vector: list[float] = Field(..., description="Concatenated fused vector (text + tabular)")
    dim: int
    preliminary_risk_score: float = Field(..., ge=0.0, le=1.0, description="Pre-fusion risk signal [0–1]")
    text_embedding: TextEmbeddingOutput
    tabular_embedding: TabularEmbeddingOutput
    metadata: dict = Field(default_factory=dict, description="Provider, model version, latency etc.")
