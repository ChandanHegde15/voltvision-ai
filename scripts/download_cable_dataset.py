"""Download only the MVTec AD cable category with the active virtualenv's API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT

REPO_ID = "foersben/mvtec-ad"

def main() -> int:
    try:
        from huggingface_hub import HfApi, snapshot_download
        files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        cable_files = [f for f in files if f.startswith("cable/")]
        if not cable_files:
            print(f"ERROR: {REPO_ID} is reachable but exposes no cable/ files.")
            return 2
        target = ROOT / "data" / "mvtec_anomaly_detection"
        target.mkdir(parents=True, exist_ok=True)
        print(f"Repository verified: {len(cable_files)} cable files. Downloading to {target} ...")
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=["cable/**"], local_dir=target)
        print("Download completed (existing cached files were reused where supported).")
        return 0
    except Exception as exc:
        print(f"ERROR: Cable subset download failed: {type(exc).__name__}: {exc}")
        print("Manual fallback: download and extract only the cable category from https://www.mvtec.com/research-teaching/datasets/mvtec-ad")
        print("Expected destination: data/mvtec_anomaly_detection/cable/")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
