import argparse, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR
from src.data_loader import image_files, test_label
from src.feature_extractor import FeatureExtractor
from src.anomaly_detector import PatchAnomalyDetector
from src.evaluation import evaluate_scores, evaluation_provenance, save_results
from src.artifacts import load_reference_artifacts

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR, help="write evaluation.json and confusion_matrix.png here"); args = parser.parse_args()
    try: bank, owners, meta = load_reference_artifacts()
    except (FileNotFoundError, RuntimeError) as exc: raise SystemExit(str(exc))
    paths=image_files(DATA_DIR/'test')
    if not paths: raise SystemExit('No MVTec cable test images found.')
    d=PatchAnomalyDetector(meta['threshold']).fit(bank, owners)
    ex=FeatureExtractor(); labels=[]; scores=[]; lat=[]
    for p in paths:
        start=time.perf_counter(); x,_=ex.extract(p); scores.append(d.score(x).score); lat.append((time.perf_counter()-start)*1000); labels.append(test_label(p))
    result=evaluate_scores(labels,scores,meta['threshold'],lat); result.update(evaluation_provenance(meta, lat))
    save_results(result, args.output_dir/'evaluation.json', args.output_dir/'confusion_matrix.png'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
