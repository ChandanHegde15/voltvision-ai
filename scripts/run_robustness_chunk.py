"""Write a bounded per-image chunk for one Experiment 4 input condition."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR
from src.anomaly_detector import PatchAnomalyDetector
from src.artifacts import load_reference_artifacts
from src.data_loader import image_files, test_label
from src.perturbations import CONDITIONS, condition_features
from src.feature_extractor import FeatureExtractor
from src.research_experiments import score_embeddings

OUT = REPORTS_DIR / "experiments" / "per_image"

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("condition", choices=CONDITIONS); parser.add_argument("start", type=int); parser.add_argument("end", type=int); args = parser.parse_args()
    paths = image_files(DATA_DIR / "test")[args.start:args.end]
    if not paths: raise SystemExit("Empty chunk.")
    bank, owners, metadata = load_reference_artifacts(); detector = PatchAnomalyDetector(0.0).fit(bank, owners); extractor = FeatureExtractor(); embeddings = [condition_features(extractor, path, args.condition) for path in paths]; scores, latencies = score_embeddings(detector, embeddings)
    rows = [{"image_id":path.relative_to(DATA_DIR).as_posix(), "category":path.parent.name, "label":test_label(path), "score":score, "latency_ms":latency} for path, score, latency in zip(paths, scores, latencies)]
    OUT.mkdir(parents=True, exist_ok=True); destination = OUT / f"robustness_{args.condition}_{args.start:03d}_{args.end:03d}.json"; destination.write_text(json.dumps({"condition":args.condition, "threshold":metadata["threshold"], "rows":rows}, indent=2), encoding="utf-8")
    print(destination)

if __name__ == "__main__": main()
