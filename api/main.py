"""FastAPI front end: a thin HTTP layer over `InspectionEngine`.

Run:  python -m uvicorn api.main:app --port 8000
Docs: http://127.0.0.1:8000/docs

Like the Streamlit app, the API starts without a dataset or artifacts. The model
is only loaded on the first `/inspect` (or `/ready`) call, and a missing or
incompatible reference bank is reported as 503, never as a crash.
"""
from __future__ import annotations

import io
import logging
import os
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from config import EVENTS_PATH
from src.edge_simulator import append_event
from src.inspection_service import InspectionEngine

log = logging.getLogger("voltvision.api")

DEVICE_ID = "API-01"
MAX_UPLOAD_BYTES = int(os.environ.get("VOLTVISION_MAX_UPLOAD_MB", "10")) * 1024 * 1024
LOG_EVENTS = os.environ.get("VOLTVISION_LOG_EVENTS", "1") != "0"

DISCLAIMER = "Visual deviation from normal reference images only. Not an electrical safety assessment; the heatmap is approximate."


class Health(BaseModel):
    status: str


class Ready(BaseModel):
    ready: bool
    detail: str | None = None
    threshold: float | None = None
    reference_images: int | None = None


class InspectionResponse(BaseModel):
    decision: str
    score: float
    threshold: float
    latency_ms: float
    heatmap: list[list[float]]
    closest_reference: str
    disclaimer: str = DISCLAIMER


@lru_cache(maxsize=1)
def get_engine() -> InspectionEngine:
    """One shared engine per process. Overridden in tests via `app.dependency_overrides`."""
    return InspectionEngine()


app = FastAPI(title="VoltVision AI", version="1.0.0", description=DISCLAIMER)


@app.get("/health", response_model=Health, summary="Liveness: the process is up (loads nothing)")
def health() -> Health:
    return Health(status="ok")


@app.get("/ready", response_model=Ready, summary="Readiness: the reference bank can be loaded and validated")
def ready(engine: InspectionEngine = Depends(get_engine)) -> Ready:
    try:
        metadata = engine.metadata
    except Exception as exc:  # missing/incompatible artifacts, missing dataset, missing weights
        raise HTTPException(status_code=503, detail=f"Inspection is not ready: {exc}") from exc
    return Ready(ready=True, threshold=metadata["threshold"], reference_images=metadata["reference_images"])


@app.post("/inspect", response_model=InspectionResponse, summary="Score one cable image")
def inspect(image: UploadFile = File(...), engine: InspectionEngine = Depends(get_engine)) -> InspectionResponse:
    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    try:
        outcome = engine.inspect(io.BytesIO(data))
    except ValueError as exc:  # unreadable / unsupported image
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inspection is not ready: {exc}") from exc

    result = outcome.result
    if LOG_EVENTS:
        try:
            append_event(EVENTS_PATH, DEVICE_ID, image.filename or "upload", result.score, result.decision, outcome.latency_ms)
        except OSError:
            log.warning("Could not append inspection event", exc_info=True)  # logging must never fail an inspection
    return InspectionResponse(
        decision=result.decision,
        score=result.score,
        threshold=outcome.metadata["threshold"],
        latency_ms=outcome.latency_ms,
        heatmap=result.heatmap.astype(float).tolist(),
        closest_reference=result.closest_reference,
    )
