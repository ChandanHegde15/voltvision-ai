"""Run the four pre-registered CPU research experiments without touching baseline artifacts."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR
from src.anomaly_detector import PatchAnomalyDetector, choose_threshold
from src.artifacts import image_id, image_manifest, load_reference_artifacts, manifest_hash
from src.data_loader import image_files, normal_training_images, test_label
from src.perturbations import CONDITIONS, condition_features
from src.feature_extractor import FeatureExtractor
from src.research_experiments import PROTOCOL, SCORING_LATENCY_SCOPE, aggregate_numeric_runs, build_bank, deterministic_split, exp04_payload, false_negative_rows, per_category_detection, report_metrics, score_embeddings

EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
SPLIT_SEEDS = list(range(10))
BASELINE_SEED = 42
BANK_SIZES = [5_000, 10_000, 20_000, 30_000]
AGGREGATION_PERCENTILES = [90.0, 95.0, 99.0]

def now() -> str: return datetime.now(timezone.utc).isoformat()
def save(name: str, payload: dict) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

def path_from_id(identifier: str) -> Path: return DATA_DIR / identifier
def ids(paths: list[Path]) -> list[str]: return [image_id(path, DATA_DIR) for path in paths]
def detector(bank: np.ndarray) -> PatchAnomalyDetector: return PatchAnomalyDetector(0.0).fit(bank, ["experiment"] * len(bank))

def extract_features(extractor: FeatureExtractor, paths: list[Path]) -> dict[Path, np.ndarray]:
    return {path: extractor.extract(path)[0] for path in paths}

def transformed_features(extractor: FeatureExtractor, paths: list[Path], condition: str, original: dict[Path, np.ndarray]) -> dict[Path, np.ndarray]:
    if condition == "identity": return original
    return {path: condition_features(extractor, path, condition) for path in paths}

def run() -> None:
    started = now(); extractor = FeatureExtractor()
    normal_paths = normal_training_images(DATA_DIR); test_paths = image_files(DATA_DIR / "test")
    labels = [test_label(path) for path in test_paths]
    categories = [path.parent.name for path in test_paths]
    reference_paths, validation_paths, _ = deterministic_split(normal_paths, BASELINE_SEED)
    features = extract_features(extractor, sorted(set(normal_paths + test_paths)))
    test_features = [features[path] for path in test_paths]
    validation_features = [features[path] for path in validation_paths]
    common = {"created_at_utc": started, "protocol": PROTOCOL, "test_images": len(test_paths), "normal_train_images": len(normal_paths), "baseline_split_seed": BASELINE_SEED, "latency_scope": SCORING_LATENCY_SCOPE}

    # Experiment 1: reference-bank size, using one nested deterministic order.
    flat = np.vstack([features[path] for path in reference_paths]); order = np.random.default_rng(2027).permutation(len(flat)); bank_runs = []
    for cap in BANK_SIZES:
        bank = flat[order[:cap]]; model = detector(bank)
        val_scores, _ = score_embeddings(model, validation_features); threshold = choose_threshold(val_scores, 99)
        scores, latencies = score_embeddings(model, test_features); metrics = report_metrics(labels, scores, threshold, latencies)
        bank_runs.append({"patch_cap": cap, "sampling_seed": 2027, "threshold": threshold, "reference_bank_bytes": int(bank.nbytes), **metrics})
    save("exp_01_reference_bank_size.json", {**common, "experiment_id": "exp_01_reference_bank_size", "changed_variable": "reference patch cap", "reference_split_hash": manifest_hash(image_manifest(reference_paths, DATA_DIR)), "validation_split_hash": manifest_hash(image_manifest(validation_paths, DATA_DIR)), "runs": bank_runs})

    # Experiment 2: score aggregation on the preserved schema-v2 normal bank.
    baseline_bank, _, baseline_metadata = load_reference_artifacts(); model = detector(baseline_bank); aggregation_runs = []
    for percentile in AGGREGATION_PERCENTILES:
        val_scores, _ = score_embeddings(model, validation_features, percentile); threshold = choose_threshold(val_scores, 99)
        scores, latencies = score_embeddings(model, test_features, percentile); metrics = report_metrics(labels, scores, threshold, latencies)
        aggregation_runs.append({"patch_score_percentile": percentile, "threshold": threshold, **metrics})
    save("exp_02_score_aggregation.json", {**common, "experiment_id": "exp_02_score_aggregation", "changed_variable": "image score aggregation percentile", "reference_artifact_format_version": baseline_metadata["artifact_format_version"], "runs": aggregation_runs})

    # Experiment 3: deterministic normal split stability.
    split_runs = []
    for seed in SPLIT_SEEDS:
        ref, val, rng = deterministic_split(normal_paths, seed); bank = build_bank(features, ref, 30_000, rng); model = detector(bank)
        val_scores, _ = score_embeddings(model, [features[path] for path in val]); threshold = choose_threshold(val_scores, 99)
        scores, latencies = score_embeddings(model, test_features); metrics = report_metrics(labels, scores, threshold, latencies)
        split_runs.append({"seed": seed, "reference_split_hash": manifest_hash(image_manifest(ref, DATA_DIR)), "validation_split_hash": manifest_hash(image_manifest(val, DATA_DIR)), "reference_images": len(ref), "validation_images": len(val), "threshold": threshold, **metrics})
    numeric = ("threshold", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate", "average_latency_ms", "median_latency_ms")
    save("exp_03_split_stability.json", {**common, "experiment_id": "exp_03_split_stability", "changed_variable": "deterministic normal train/validation split seed", "seeds": SPLIT_SEEDS, "runs": split_runs, "aggregate": aggregate_numeric_runs(split_runs, numeric)})

    # Experiment 4: frozen baseline failure and single-factor input-condition study.
    baseline_threshold = baseline_metadata["threshold"]; model = detector(baseline_bank); robustness = []
    for condition in CONDITIONS:
        condition_features = transformed_features(extractor, test_paths, condition, features); scores, latencies = score_embeddings(model, [condition_features[path] for path in test_paths]); metrics = report_metrics(labels, scores, baseline_threshold, latencies)
        robustness.append({"condition": condition, "threshold_recalibrated": False, **metrics})
        if condition == "identity":
            per_category = per_category_detection(categories, labels, scores, baseline_threshold)
            identity_rows = [{"image_id": image_id_, "category": category, "label": label, "score": score} for image_id_, category, label, score in zip(ids(test_paths), categories, labels, scores)]
            false_negatives = false_negative_rows(identity_rows, baseline_threshold)
    extra = {"normal_train_images": len(normal_paths), "baseline_split_seed": BASELINE_SEED}
    save("exp_04_robustness_failure_analysis.json", exp04_payload(created_at_utc=started, baseline_threshold=baseline_threshold, runs=robustness, per_category=per_category, false_negatives=false_negatives, test_images=len(test_paths), extra=extra))
    print(json.dumps({"completed_at_utc": now(), "outputs": [str(EXPERIMENTS_DIR / name) for name in ("exp_01_reference_bank_size.json", "exp_02_score_aggregation.json", "exp_03_split_stability.json", "exp_04_robustness_failure_analysis.json")]}, indent=2))

if __name__ == "__main__": run()
