"""Aggregate completed Experiment 4 per-image chunks into the single Exp 4 report.

Reads only existing chunk/summary JSON and never runs a model. Use --output to
write a candidate file and compare it before replacing the recorded report.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPORTS_DIR
from src.research_experiments import EXP04_ID, aggregate_exp04

ROOT = REPORTS_DIR / "experiments"; CHUNKS = ROOT / "per_image"; OUTPUT = ROOT / f"{EXP04_ID}.json"
EXPECTED_TEST_IMAGES = 150  # MVTec AD cable test split

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUTPUT); args = parser.parse_args()
    existing = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    try: payload = aggregate_exp04(CHUNKS, existing, EXPECTED_TEST_IMAGES, datetime.now(timezone.utc).isoformat())
    except ValueError as exc: raise SystemExit(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)

if __name__ == "__main__": main()
