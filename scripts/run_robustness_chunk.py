"""Write a bounded per-image chunk for one Experiment 4 input condition."""
from __future__ import annotations
import argparse, io, json
from pathlib import Path
import sys
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR
from src.anomaly_detector import PatchAnomalyDetector
from src.artifacts import load_reference_artifacts
from src.data_loader import image_files
from src.feature_extractor import FeatureExtractor
from src.research_experiments import score_embeddings

CONDITIONS = ("identity", "blur_0_5", "blur_1_0", "brightness_0_8", "brightness_1_2")
OUT = REPORTS_DIR / "experiments" / "per_image"

def transformed(extractor: FeatureExtractor, path: Path, condition: str):
    if condition == "identity": return extractor.extract(path)[0]
    with Image.open(path) as source: image = source.convert("RGB")
    if condition == "blur_0_5": image = image.filter(ImageFilter.GaussianBlur(.5))
    elif condition == "blur_1_0": image = image.filter(ImageFilter.GaussianBlur(1.0))
    elif condition == "brightness_0_8": image = ImageEnhance.Brightness(image).enhance(.8)
    elif condition == "brightness_1_2": image = ImageEnhance.Brightness(image).enhance(1.2)
    stream = io.BytesIO(); image.save(stream, format="PNG"); stream.seek(0); return extractor.extract(stream)[0]

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("condition", choices=CONDITIONS); parser.add_argument("start", type=int); parser.add_argument("end", type=int); args = parser.parse_args()
    paths = image_files(DATA_DIR / "test")[args.start:args.end]
    if not paths: raise SystemExit("Empty chunk.")
    bank, owners, metadata = load_reference_artifacts(); detector = PatchAnomalyDetector(0.0).fit(bank, owners); extractor = FeatureExtractor(); embeddings = [transformed(extractor, path, args.condition) for path in paths]; scores, latencies = score_embeddings(detector, embeddings)
    rows = [{"image_id":path.relative_to(DATA_DIR).as_posix(), "category":path.parent.name, "label":0 if path.parent.name == "good" else 1, "score":score, "latency_ms":latency} for path, score, latency in zip(paths, scores, latencies)]
    OUT.mkdir(parents=True, exist_ok=True); destination = OUT / f"robustness_{args.condition}_{args.start:03d}_{args.end:03d}.json"; destination.write_text(json.dumps({"condition":args.condition, "threshold":metadata["threshold"], "rows":rows}, indent=2), encoding="utf-8")
    print(destination)

if __name__ == "__main__": main()
