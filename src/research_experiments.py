"""Reusable, deterministic utilities for controlled VoltVision experiments."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
from src.anomaly_detector import PatchAnomalyDetector
from src.evaluation import evaluate_scores

def deterministic_split(paths: list[Path], seed: int, validation_fraction: float = 0.2) -> tuple[list[Path], list[Path], np.random.Generator]:
    """Return deterministic 80/20 normal reference/validation paths and RNG state."""
    ordered = sorted(paths); rng = np.random.default_rng(seed); indices = rng.permutation(len(ordered)); validation_size = max(1, round(len(ordered) * validation_fraction)); validation_indices = {int(index) for index in indices[:validation_size]}
    return [path for index, path in enumerate(ordered) if index not in validation_indices], [path for index, path in enumerate(ordered) if index in validation_indices], rng

def build_bank(features: dict[Path, np.ndarray], reference_paths: list[Path], cap: int, rng: np.random.Generator) -> np.ndarray:
    patches = np.vstack([features[path] for path in reference_paths])
    return patches if len(patches) <= cap else patches[rng.choice(len(patches), cap, replace=False)]

def score_embeddings(detector: PatchAnomalyDetector, features: Iterable[np.ndarray], percentile: float = 95.0) -> tuple[list[float], list[float]]:
    import time
    scores: list[float] = []; latencies: list[float] = []
    for patches in features:
        started = time.perf_counter(); scores.append(detector.anomaly_score(patches, percentile)); latencies.append((time.perf_counter() - started) * 1000)
    return scores, latencies

def report_metrics(labels: list[int], scores: list[float], threshold: float, latencies: list[float]) -> dict:
    report = evaluate_scores(labels, scores, threshold, latencies); tn, fp, fn, tp = np.asarray(report["confusion_matrix"]).ravel()
    report.update({"true_negatives":int(tn), "false_positives":int(fp), "false_negatives":int(fn), "true_positives":int(tp), "false_positive_rate":float(fp/(fp+tn)) if fp+tn else None, "false_negative_rate":float(fn/(fn+tp)) if fn+tp else None, "median_latency_ms":float(np.median(latencies)) if latencies else None})
    return report

def aggregate_numeric_runs(runs: list[dict], fields: tuple[str, ...]) -> dict:
    return {field:{"mean":float(np.mean([run[field] for run in runs])), "std":float(np.std([run[field] for run in runs],ddof=0)), "min":float(np.min([run[field] for run in runs])), "max":float(np.max([run[field] for run in runs]))} for field in fields}

# ---- Experiment 4: one schema shared by every writer ----------------------------
EXP04_ID = "exp_04_robustness_failure_analysis"
PROTOCOL = "pre-registered exploratory comparison; test labels are used only for final reporting"
SCORING_LATENCY_SCOPE = "nearest-neighbor scoring only; feature extraction is precomputed and excluded"
EXP04_LIMITATIONS = "Synthetic conditions are not a substitute for real production camera and lighting variation."

def per_category_detection(categories: list[str], labels: list[int], scores: list[float], threshold: float) -> dict:
    """Per-category outcome at a fixed threshold.

    Defect categories contain only anomalous images, so precision and ROC-AUC are
    undefined there; only the detection rate (recall) is meaningful. The good
    category reports normal acceptance instead.
    """
    report: dict[str, dict] = {}
    for category in sorted(set(categories)):
        idx = [i for i, value in enumerate(categories) if value == category]
        flagged = int(sum(scores[i] >= threshold for i in idx))
        if all(labels[i] == 0 for i in idx):
            report[category] = {"images": len(idx), "false_positives": flagged, "normal_acceptance_rate": 1.0 - flagged / len(idx)}
        elif all(labels[i] == 1 for i in idx):
            report[category] = {"images": len(idx), "detected": flagged, "detection_rate": flagged / len(idx)}
        else:
            raise ValueError(f"Category {category!r} mixes normal and anomalous labels.")
    return report

def detection_from_legacy_category(category: str, entry: dict) -> dict:
    """Convert a per-category entry written by earlier Exp 4 scripts to the detection schema."""
    if "detection_rate" in entry or ("normal_acceptance_rate" in entry and "false_positives" in entry):
        return dict(entry)
    if "true_positives" not in entry:  # legacy hand-written normal-only entry
        images = int(entry["images"]); false_positives = round(entry["false_positive_rate"] * images)
        return {"images": images, "false_positives": false_positives, "normal_acceptance_rate": 1.0 - false_positives / images}
    positives = entry["true_positives"] + entry["false_negatives"]
    if positives == 0:
        images = entry["true_negatives"] + entry["false_positives"]
        return {"images": images, "false_positives": entry["false_positives"], "normal_acceptance_rate": 1.0 - entry["false_positives"] / images}
    if entry["true_negatives"] + entry["false_positives"]: raise ValueError(f"Category {category!r} mixes normal and anomalous labels.")
    return {"images": positives, "detected": entry["true_positives"], "detection_rate": entry["true_positives"] / positives}

def false_negative_rows(rows: list[dict], threshold: float) -> list[dict]:
    return [{"image_id": row["image_id"], "category": row["category"], "score": row["score"], "margin_to_threshold": threshold - row["score"]} for row in rows if row["label"] == 1 and row["score"] < threshold]

def exp04_payload(*, created_at_utc: str, baseline_threshold: float, runs: list[dict], per_category: dict, false_negatives: list[dict], test_images: int, extra: dict | None = None) -> dict:
    return {"experiment_id": EXP04_ID, "created_at_utc": created_at_utc, "protocol": PROTOCOL, "latency_scope": SCORING_LATENCY_SCOPE, "changed_variable": "input quality condition", "test_images": int(test_images), "baseline_threshold": float(baseline_threshold), "threshold_recalibrated": False, "runs": runs, "per_category_original": per_category, "false_negatives_original": false_negatives, "limitations": EXP04_LIMITATIONS, **(extra or {})}

def load_condition_rows(chunk_dir: Path, condition: str, expected_rows: int) -> tuple[list[dict], float] | None:
    """Merge per-image chunks for one condition; None when the condition has no chunks.

    Overlapping chunk ranges are allowed, but a duplicated image must carry the same
    score, and every chunk must use the same threshold.
    """
    import json
    files = sorted(Path(chunk_dir).glob(f"robustness_{condition}_*.json"))
    if not files: return None
    rows: dict[str, dict] = {}; thresholds = set()
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8")); thresholds.add(payload["threshold"])
        for row in payload["rows"]:
            previous = rows.get(row["image_id"])
            if previous is not None and previous["score"] != row["score"]: raise ValueError(f"{condition}: conflicting scores for {row['image_id']} across chunks.")
            rows[row["image_id"]] = row  # last chunk in name order wins (only its latency can differ)
    if len(thresholds) != 1: raise ValueError(f"{condition}: chunks use different thresholds {sorted(thresholds)}.")
    if len(rows) != expected_rows: raise ValueError(f"{condition}: expected {expected_rows} unique rows, found {len(rows)}. Run missing chunks before aggregation.")
    return [rows[key] for key in sorted(rows)], thresholds.pop()

def aggregate_exp04(chunk_dir: Path, existing: dict, expected_rows: int, created_at_utc: str) -> dict:
    """Build the Exp 4 report from per-image chunks, reusing the existing identity
    summary when no identity chunks exist. Never runs a model."""
    from src.perturbations import CONDITIONS
    runs: list[dict] = []; identity_rows = None; thresholds = set()
    for condition in CONDITIONS:
        loaded = load_condition_rows(chunk_dir, condition, expected_rows)
        if loaded is None:
            if condition != "identity": raise ValueError(f"{condition}: no chunks available.")
            prior = next((run for run in existing.get("runs", []) if run.get("condition") == "identity"), None)
            if prior is None: raise ValueError("identity: no chunks and no prior identity summary available.")
            runs.append(prior); thresholds.add(prior["threshold"]); continue
        rows, threshold = loaded; thresholds.add(threshold)
        metrics = report_metrics([row["label"] for row in rows], [row["score"] for row in rows], threshold, [row["latency_ms"] for row in rows])
        runs.append({"condition": condition, "threshold_recalibrated": False, **metrics})
        if condition == "identity": identity_rows = rows
    if len(thresholds) != 1: raise ValueError(f"Conditions use different thresholds {sorted(thresholds)}.")
    threshold = thresholds.pop()
    if identity_rows is not None:
        per_category = per_category_detection([r["category"] for r in identity_rows], [r["label"] for r in identity_rows], [r["score"] for r in identity_rows], threshold)
        false_negatives = false_negative_rows(identity_rows, threshold)
    else:
        per_category = {name: detection_from_legacy_category(name, entry) for name, entry in existing.get("per_category_original", {}).items()}
        false_negatives = existing.get("false_negatives_original", [])
    return exp04_payload(created_at_utc=created_at_utc, baseline_threshold=threshold, runs=runs, per_category=per_category, false_negatives=false_negatives, test_images=expected_rows)
