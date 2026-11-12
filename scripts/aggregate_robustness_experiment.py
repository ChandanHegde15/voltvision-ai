"""Aggregate all completed Experiment 4 per-image chunks."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPORTS_DIR
from src.perturbations import CONDITIONS
from src.research_experiments import report_metrics

ROOT = REPORTS_DIR / "experiments"; CHUNKS = ROOT / "per_image"; OUTPUT = ROOT / "exp_04_robustness_failure_analysis.json"

def main() -> None:
    existing = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    runs = []; identity_rows = []
    for condition in CONDITIONS:
        if condition == "identity" and not list(CHUNKS.glob("robustness_identity_*.json")):
            prior = next((run for run in existing.get("runs", []) if run.get("condition") == "identity"), None)
            if prior is None: raise SystemExit("identity: no completed summary or chunks available.")
            runs.append(prior)
            continue
        rows = []
        for file in sorted(CHUNKS.glob(f"robustness_{condition}_*.json")):
            rows.extend(json.loads(file.read_text())["rows"])
        dedup = {row["image_id"]: row for row in rows}
        if len(dedup) != 150: raise SystemExit(f"{condition}: expected 150 unique rows, found {len(dedup)}. Run missing chunks before aggregation.")
        rows = [dedup[key] for key in sorted(dedup)]; threshold = json.loads(next(CHUNKS.glob(f"robustness_{condition}_*.json")).read_text())["threshold"]
        metrics = report_metrics([row["label"] for row in rows], [row["score"] for row in rows], threshold, [row["latency_ms"] for row in rows]); runs.append({"condition":condition, "threshold_recalibrated":False, **metrics})
        if condition == "identity": identity_rows = rows
    if not identity_rows:
        per_category = existing.get("per_category_original", {})
        identity = next(run for run in runs if run["condition"] == "identity")
        per_category["good"] = {"images": identity["true_negatives"] + identity["false_positives"], "normal_acceptance_rate": 1.0 - identity["false_positive_rate"], "false_positive_rate": identity["false_positive_rate"], "note": "Anomaly recall is undefined because this group contains no positive/anomalous labels."}
        OUTPUT.write_text(json.dumps({"experiment_id":"exp_04_robustness_failure_analysis", "created_at_utc":datetime.now(timezone.utc).isoformat(), "changed_variable":"input quality condition", "baseline_threshold":threshold, "runs":runs, "per_category_original":per_category, "false_negatives_original":existing.get("false_negatives_original", []), "limitations":"Synthetic conditions are not a substitute for real production camera and lighting variation."}, indent=2), encoding="utf-8")
        print(OUTPUT); return
    categories = sorted({row["category"] for row in identity_rows}); per_category = {}
    for category in categories:
        subset = [row for row in identity_rows if row["category"] == category]; per_category[category] = report_metrics([row["label"] for row in subset], [row["score"] for row in subset], threshold, [row["latency_ms"] for row in subset])
    false_negatives = [{"image_id":row["image_id"], "category":row["category"], "score":row["score"], "margin_to_threshold":threshold-row["score"]} for row in identity_rows if row["label"] == 1 and row["score"] < threshold]
    OUTPUT.write_text(json.dumps({"experiment_id":"exp_04_robustness_failure_analysis", "created_at_utc":datetime.now(timezone.utc).isoformat(), "changed_variable":"input quality condition", "baseline_threshold":threshold, "runs":runs, "per_category_original":per_category, "false_negatives_original":false_negatives, "limitations":"Synthetic conditions are not a substitute for real production camera and lighting variation."}, indent=2), encoding="utf-8")
    print(OUTPUT)

if __name__ == "__main__": main()
