"""Resumable single-condition runner for Experiment 4 under short process limits."""
from __future__ import annotations
import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR
from src.anomaly_detector import PatchAnomalyDetector
from src.artifacts import load_reference_artifacts
from src.data_loader import image_files
from src.feature_extractor import FeatureExtractor
from src.research_experiments import report_metrics, score_embeddings

OUTPUT = REPORTS_DIR / "experiments" / "exp_04_robustness_failure_analysis.json"
CONDITIONS = ("identity", "blur_0_5", "blur_1_0", "brightness_0_8", "brightness_1_2")

def features_for_condition(extractor: FeatureExtractor, paths: list[Path], condition: str):
    output = []
    for path in paths:
        if condition == "identity":
            output.append(extractor.extract(path)[0]); continue
        with Image.open(path) as source: image = source.convert("RGB")
        if condition == "blur_0_5": image = image.filter(ImageFilter.GaussianBlur(radius=.5))
        elif condition == "blur_1_0": image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif condition == "brightness_0_8": image = ImageEnhance.Brightness(image).enhance(.8)
        elif condition == "brightness_1_2": image = ImageEnhance.Brightness(image).enhance(1.2)
        stream = io.BytesIO(); image.save(stream, format="PNG"); stream.seek(0); output.append(extractor.extract(stream)[0])
    return output

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("condition", choices=CONDITIONS); condition = parser.parse_args().condition
    bank, owners, metadata = load_reference_artifacts(); detector = PatchAnomalyDetector(0.0).fit(bank, owners); extractor = FeatureExtractor(); paths = image_files(DATA_DIR / "test"); labels = [0 if path.parent.name == "good" else 1 for path in paths]
    scores, latencies = score_embeddings(detector, features_for_condition(extractor, paths, condition)); report = report_metrics(labels, scores, metadata["threshold"], latencies)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {"experiment_id":"exp_04_robustness_failure_analysis", "created_at_utc":datetime.now(timezone.utc).isoformat(), "changed_variable":"input quality condition", "baseline_threshold":metadata["threshold"], "runs":[], "limitations":"Synthetic conditions are not a substitute for real production camera and lighting variation."}
    payload["runs"] = [run for run in payload["runs"] if run["condition"] != condition] + [{"condition":condition, "threshold_recalibrated":False, **report}]
    payload["runs"].sort(key=lambda run: CONDITIONS.index(run["condition"]))
    if condition == "identity":
        categories = [path.parent.name for path in paths]; per_category = {}; false_negatives = []
        for category in sorted(set(categories)):
            indexes = [i for i, value in enumerate(categories) if value == category]
            per_category[category] = report_metrics([labels[i] for i in indexes], [scores[i] for i in indexes], metadata["threshold"], [latencies[i] for i in indexes])
        for i, label in enumerate(labels):
            if label == 1 and scores[i] < metadata["threshold"]: false_negatives.append({"image_id":paths[i].relative_to(DATA_DIR).as_posix(), "category":categories[i], "score":scores[i], "margin_to_threshold":metadata["threshold"]-scores[i]})
        payload["per_category_original"] = per_category; payload["false_negatives_original"] = false_negatives
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"condition":condition,"output":str(OUTPUT),"completed_runs":[run["condition"] for run in payload["runs"]]}, indent=2))

if __name__ == "__main__": main()
