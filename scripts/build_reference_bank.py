import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, ARTIFACTS_DIR, SEED, REFERENCE_CAP, WEIGHTS_ID, IMAGE_SIZE, FEATURE_LAYER
from src.data_loader import normal_training_images
from src.feature_extractor import FeatureExtractor
from src.anomaly_detector import PatchAnomalyDetector, choose_threshold
from src.artifacts import image_id, image_manifest, make_metadata
from src.research_experiments import deterministic_split

def main():
    paths = normal_training_images(DATA_DIR)
    if len(paths) < 4: raise SystemExit(f"Need at least 4 readable train/good images at {DATA_DIR}; found {len(paths)}.")
    # Same seeded 80/20 split as the experiments; the returned RNG continues into patch sampling.
    ref, val, rng = deterministic_split(paths, SEED)
    ex=FeatureExtractor(); chunks=[]; owners=[]
    for p in ref:
        x,_=ex.extract(p); chunks.append(x); owners.extend([image_id(p, DATA_DIR)]*len(x))
    bank=np.vstack(chunks)
    if len(bank)>REFERENCE_CAP:
        chosen=rng.choice(len(bank),REFERENCE_CAP,replace=False)
        bank=bank[chosen]; owners=[owners[int(i)] for i in chosen]
    if set(owners) != {image_id(path, DATA_DIR) for path in ref}:
        raise RuntimeError("Reference patch sampling omitted a source image. Increase REFERENCE_CAP and rebuild.")
    provisional=PatchAnomalyDetector(0).fit(bank,owners)
    threshold=choose_threshold([provisional.score(ex.extract(p)[0]).score for p in val],99)
    ARTIFACTS_DIR.mkdir(exist_ok=True); np.save(ARTIFACTS_DIR/'reference_patches.npy',bank)
    (ARTIFACTS_DIR/'reference_paths.json').write_text(json.dumps(owners),encoding='utf-8')
    meta=make_metadata(reference_manifest=image_manifest(ref, DATA_DIR), validation_manifest=image_manifest(val, DATA_DIR), threshold=threshold, reference_patch_count=len(bank), dataset_root=DATA_DIR)
    (ARTIFACTS_DIR/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
