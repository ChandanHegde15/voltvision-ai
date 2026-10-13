from __future__ import annotations
from collections import Counter
from pathlib import Path
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if folder.exists() else []

def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False

def inspect_dataset(root: Path) -> dict:
    root = Path(root)
    expected = [root / "train" / "good", root / "test", root / "ground_truth"]
    train = image_files(root / "train" / "good")
    test_root = root / "test"
    tests = image_files(test_root)
    test_categories = Counter(p.parent.name for p in tests)
    masks = image_files(root / "ground_truth")
    invalid = [str(p) for p in train + tests if not validate_image(p)]
    return {"exists": root.exists(), "root": str(root), "train_good": len(train),
            "test_total": len(tests), "test_by_category": dict(test_categories),
            "mask_count": len(masks), "total": len(train) + len(tests),
            "missing": [str(p) for p in expected if not p.exists()], "invalid": invalid,
            "ready": bool(train) and bool(tests) and not invalid}

def normal_training_images(root: Path) -> list[Path]:
    return [p for p in image_files(Path(root) / "train" / "good") if validate_image(p)]
