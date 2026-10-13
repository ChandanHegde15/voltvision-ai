from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

EVENT_COLUMNS = ["timestamp_utc", "device_id", "image_filename", "anomaly_score", "decision", "latency_ms", "alert"]
def append_event(path: Path, device_id: str, filename: str, score: float, decision: str, latency_ms: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "device_id": device_id, "image_filename": filename,
           "anomaly_score": float(score), "decision": decision, "latency_ms": float(latency_ms), "alert": decision == "REVIEW REQUIRED"}
    pd.DataFrame([row], columns=EVENT_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)
def load_events(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=EVENT_COLUMNS)
