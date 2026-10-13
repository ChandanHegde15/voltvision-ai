import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, ARTIFACTS_DIR, SEED, REFERENCE_CAP, WEIGHTS_ID, IMAGE_SIZE, FEATURE_LAYER
from src.data_loader import normal_training_images
from src.feature_extractor import FeatureExtractor
from src.anomaly_detector import PatchAnomalyDetector, choose_threshold

def main():
    paths = normal_training_images(DATA_DIR)
    if len(paths) < 4: raise SystemExit(f"Need at least 4 readable train/good images at {DATA_DIR}; found {len(paths)}.")
    rng=np.random.default_rng(SEED); ids=rng.permutation(len(paths)); cut=max(1,round(len(paths)*.2)); validation={int(i) for i in ids[:cut]}
    ref=[p for i,p in enumerate(paths) if i not in validation]; val=[p for i,p in enumerate(paths) if i in validation]
    ex=FeatureExtractor(); chunks=[]; owners=[]
    for p in ref:
        x,_=ex.extract(p); chunks.append(x); owners.extend([str(p)]*len(x))
    bank=np.vstack(chunks)
    if len(bank)>REFERENCE_CAP:
        chosen=rng.choice(len(bank),REFERENCE_CAP,replace=False)
        bank=bank[chosen]; owners=[owners[int(i)] for i in chosen]
    provisional=PatchAnomalyDetector(0).fit(bank,owners)
    threshold=choose_threshold([provisional.score(ex.extract(p)[0]).score for p in val],99)
    ARTIFACTS_DIR.mkdir(exist_ok=True); np.save(ARTIFACTS_DIR/'reference_patches.npy',bank)
    (ARTIFACTS_DIR/'reference_paths.json').write_text(json.dumps(owners),encoding='utf-8')
    meta={'architecture':'resnet18','weights':WEIGHTS_ID,'image_size':IMAGE_SIZE,'feature_layer':FEATURE_LAYER,'metric':'euclidean','threshold':threshold,'threshold_method':'99th percentile of held-out normal training validation split','reference_images':len(ref),'validation_images':len(val),'dataset':str(DATA_DIR)}
    (ARTIFACTS_DIR/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
