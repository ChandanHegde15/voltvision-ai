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
