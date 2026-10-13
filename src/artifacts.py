from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from config import ARTIFACTS_DIR, FEATURE_LAYER, IMAGE_SIZE, WEIGHTS_ID

REQUIRED = ("metadata.json", "reference_patches.npy", "reference_paths.json")

def load_reference_artifacts(root: Path = ARTIFACTS_DIR) -> tuple[np.ndarray, list[str], dict]:
    root = Path(root)
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Reference bank is unavailable (missing " + ", ".join(missing) + "). Run python scripts/build_reference_bank.py.")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        patches = np.load(root / "reference_patches.npy")
        paths = json.loads((root / "reference_paths.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Reference artifacts are corrupt: {exc}") from exc
    expected = {"architecture": "resnet18", "weights": WEIGHTS_ID, "image_size": IMAGE_SIZE, "feature_layer": FEATURE_LAYER, "metric": "euclidean"}
    mismatch = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatch or patches.ndim != 2 or len(patches) < 2 or len(paths) != len(patches):
        detail = ", ".join(mismatch) if mismatch else "invalid patch or path arrays"
        raise RuntimeError(f"Reference artifacts are incompatible ({detail}). Rebuild with python scripts/build_reference_bank.py.")
    return patches.astype(np.float32, copy=False), paths, metadata
