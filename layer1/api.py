"""
api.py

FastAPI application — exposes Layer 1 as a REST endpoint.

POST /layer1/process  →  FusedVector JSON

Run locally:
    uvicorn layer1.api:app --reload --port 8000

Then test:
    curl -X POST http://localhost:8000/layer1/process \
      -H "Content-Type: application/json" \
      -d '{
        "user_id": "user_001",
        "message": "I haven't been sleeping well and feel really isolated lately",
        "logs": {"sleep_hours": 4.5, "mood_score": 3, "social_score": 2}
      }'
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from layer1.layer1 import Layer1Pipeline
from layer1.models.schemas import Layer1Input, FusedVector

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ── App state ─────────────────────────────────────────────────────────────────

pipeline: Layer1Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise pipeline on startup, clean up on shutdown."""
    global pipeline
    logger.info("Initialising Layer 1 pipeline...")
    pipeline = Layer1Pipeline.default(
        tabular_model_path=os.getenv("TABULAR_MODEL_PATH"),
        fusion_model_path=os.getenv("FUSION_MODEL_PATH"),
        device=os.getenv("TORCH_DEVICE", "cpu"),
    )
    logger.info("Layer 1 pipeline ready.")
    yield
    logger.info("Shutting down Layer 1 pipeline.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Augur — Layer 1: Sensory & Early Fusion",
    description=(
        "Ingests user chat messages and self-report logs, "
        "produces a fused vector and preliminary risk score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.post("/layer1/process", response_model=FusedVector)
async def process(input_data: Layer1Input) -> FusedVector:
    """
    Run the full Layer 1 pipeline:
      1. Gemini text embedding
      2. FT-Transformer tabular encoding
      3. Gated fusion → FusedVector

    The returned FusedVector is consumed by:
      - Layer 3 safety guardrails (preliminary_risk_score)
      - Layer 2 Kenko dialogue engine
      - Layer 4 insight extraction engine
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    try:
        return await pipeline.process(input_data)
    except Exception as e:
        logger.exception("Layer1 pipeline error for user=%s: %s", input_data.user_id, e)
        raise HTTPException(status_code=500, detail="Internal pipeline error")
