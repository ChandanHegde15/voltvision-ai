import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, REPORTS_DIR
from src.data_loader import image_files
from src.feature_extractor import FeatureExtractor
from src.anomaly_detector import PatchAnomalyDetector
from src.evaluation import evaluate_scores, save_results
from src.artifacts import load_reference_artifacts

def main():
    try: bank, owners, meta = load_reference_artifacts()
    except (FileNotFoundError, RuntimeError) as exc: raise SystemExit(str(exc))
    paths=image_files(DATA_DIR/'test')
    if not paths: raise SystemExit('No MVTec cable test images found.')
    d=PatchAnomalyDetector(meta['threshold']).fit(bank, owners)
    ex=FeatureExtractor(); labels=[]; scores=[]; lat=[]
    for p in paths:
        start=time.perf_counter(); x,_=ex.extract(p); scores.append(d.score(x).score); lat.append((time.perf_counter()-start)*1000); labels.append(0 if p.parent.name=='good' else 1)
    result=evaluate_scores(labels,scores,meta['threshold'],lat); result['threshold_selection_method']=meta['threshold_method']
    save_results(result, REPORTS_DIR/'evaluation.json', REPORTS_DIR/'confusion_matrix.png'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
