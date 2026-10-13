import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR
from src.data_loader import inspect_dataset
report = inspect_dataset(DATA_DIR)
print(json.dumps(report, indent=2))
if not report["ready"]:
    print("\nDataset unavailable or incomplete. Run: .\\.venv\\Scripts\\python.exe scripts\\download_cable_dataset.py")
    print("Manual fallback: https://www.mvtec.com/research-teaching/datasets/mvtec-ad")
    raise SystemExit(2)
