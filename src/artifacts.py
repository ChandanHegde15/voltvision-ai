"""Versioned reference-bank metadata and compatibility checks."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable
import numpy as np
from config import ARTIFACT_FORMAT_VERSION, ARTIFACTS_DIR, DATA_DIR, FEATURE_LAYER, IMAGE_SIZE, PREPROCESSING_CONFIG, REFERENCE_CAP, SEED, WEIGHTS_ID
from src.data_loader import image_files

REQUIRED = ("metadata.json", "reference_patches.npy", "reference_paths.json")
REQUIRED_METADATA = {"artifact_format_version", "created_at_utc", "seed", "reference_patch_cap", "reference_patch_count", "architecture", "weights", "image_size", "feature_layer", "preprocessing", "metric", "threshold", "threshold_method", "reference_images", "validation_images", "dataset_path", "dataset_fingerprint", "reference_image_manifest", "validation_image_manifest", "reference_image_list_hash", "validation_image_list_hash"}

def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def image_id(path: Path, dataset_root: Path) -> str:
    """Stable, portable dataset-relative POSIX image identifier."""
    return Path(path).resolve().relative_to(Path(dataset_root).resolve()).as_posix()

def image_manifest(paths: Iterable[Path], dataset_root: Path) -> list[dict[str, str]]:
    """Sorted source-content manifest for deterministic split verification."""
    return sorted(({"id": image_id(path, dataset_root), "sha256": _sha256_file(path)} for path in paths), key=lambda item: item["id"])

def manifest_hash(manifest: list[dict[str, str]]) -> str:
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()

def dataset_fingerprint(dataset_root: Path) -> str:
    """Content fingerprint for every supported image and mask under cable/."""
    return manifest_hash(image_manifest(image_files(dataset_root), dataset_root))

def make_metadata(*, reference_manifest: list[dict[str, str]], validation_manifest: list[dict[str, str]], threshold: float, reference_patch_count: int, dataset_root: Path = DATA_DIR) -> dict:
    return {"artifact_format_version": ARTIFACT_FORMAT_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "reference_patch_cap": REFERENCE_CAP, "reference_patch_count": int(reference_patch_count), "architecture": "resnet18", "weights": WEIGHTS_ID, "image_size": IMAGE_SIZE, "feature_layer": FEATURE_LAYER, "preprocessing": PREPROCESSING_CONFIG, "metric": "euclidean", "threshold": float(threshold), "threshold_method": "99th percentile of held-out normal training validation split", "reference_images": len(reference_manifest), "validation_images": len(validation_manifest), "dataset_path": str(Path(dataset_root).resolve()), "dataset_fingerprint": dataset_fingerprint(dataset_root), "reference_image_manifest": reference_manifest, "validation_image_manifest": validation_manifest, "reference_image_list_hash": manifest_hash(reference_manifest), "validation_image_list_hash": manifest_hash(validation_manifest)}

def _incompatible(detail: str) -> RuntimeError:
    return RuntimeError(f"Reference artifacts are incompatible ({detail}). Rebuild with python scripts/build_reference_bank.py.")

def validate_metadata(metadata: dict, dataset_root: Path = DATA_DIR) -> None:
    missing = sorted(REQUIRED_METADATA - metadata.keys())
    if missing: raise _incompatible("missing metadata: " + ", ".join(missing))
    expected = {"artifact_format_version": ARTIFACT_FORMAT_VERSION, "seed": SEED, "reference_patch_cap": REFERENCE_CAP, "architecture": "resnet18", "weights": WEIGHTS_ID, "image_size": IMAGE_SIZE, "feature_layer": FEATURE_LAYER, "preprocessing": PREPROCESSING_CONFIG, "metric": "euclidean", "dataset_path": str(Path(dataset_root).resolve())}
    mismatch = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatch: raise _incompatible("configuration mismatch: " + ", ".join(mismatch))
    for name in ("reference_image_manifest", "validation_image_manifest"):
        manifest = metadata[name]
        if not isinstance(manifest, list) or not all(isinstance(item, dict) and set(item) == {"id", "sha256"} for item in manifest): raise _incompatible(f"invalid {name}")
    if len(metadata["reference_image_manifest"]) != metadata["reference_images"] or len(metadata["validation_image_manifest"]) != metadata["validation_images"]: raise _incompatible("manifest image counts do not match metadata")
    if manifest_hash(metadata["reference_image_manifest"]) != metadata["reference_image_list_hash"]: raise _incompatible("reference image-list hash mismatch")
    if manifest_hash(metadata["validation_image_manifest"]) != metadata["validation_image_list_hash"]: raise _incompatible("validation image-list hash mismatch")
    try: current_fingerprint = dataset_fingerprint(dataset_root)
    except OSError as exc: raise _incompatible(f"cannot fingerprint dataset: {exc}") from exc
    if current_fingerprint != metadata["dataset_fingerprint"]: raise _incompatible("dataset fingerprint mismatch")
    for name in ("reference_image_manifest", "validation_image_manifest"):
        current = image_manifest([Path(dataset_root) / item["id"] for item in metadata[name]], dataset_root)
        if current != metadata[name]: raise _incompatible(f"{name} no longer matches dataset contents")

def load_reference_artifacts(root: Path = ARTIFACTS_DIR, dataset_root: Path = DATA_DIR) -> tuple[np.ndarray, list[str], dict]:
    root = Path(root); missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing: raise FileNotFoundError("Reference bank is unavailable (missing " + ", ".join(missing) + "). Run python scripts/build_reference_bank.py.")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8")); patches = np.load(root / "reference_patches.npy"); paths = json.loads((root / "reference_paths.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc: raise RuntimeError(f"Reference artifacts are corrupt: {exc}") from exc
    validate_metadata(metadata, dataset_root); allowed = {item["id"] for item in metadata["reference_image_manifest"]}
    if patches.ndim != 2 or patches.shape[1] != 256 or not np.isfinite(patches).all() or len(patches) < 2: raise _incompatible("invalid reference patch array")
    if len(paths) != len(patches) or not all(isinstance(path, str) and path in allowed for path in paths): raise _incompatible("invalid reference patch ownership mapping")
    return patches.astype(np.float32, copy=False), paths, metadata
