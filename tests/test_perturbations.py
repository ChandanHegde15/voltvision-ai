from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter
from src.data_loader import test_label
from src.perturbations import CONDITIONS, apply_condition
from src.research_experiments import deterministic_split

def _legacy_condition(image: Image.Image, condition: str) -> Image.Image:
    """Inline perturbation code as it existed in the experiment scripts before consolidation."""
    if condition == "blur_0_5": return image.filter(ImageFilter.GaussianBlur(radius=0.5))
    if condition == "blur_1_0": return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    if condition == "brightness_0_8": return ImageEnhance.Brightness(image).enhance(0.8)
    if condition == "brightness_1_2": return ImageEnhance.Brightness(image).enhance(1.2)
    return image

def _fixed_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8), "RGB")

def test_conditions_are_the_preregistered_set():
    assert CONDITIONS == ("identity", "blur_0_5", "blur_1_0", "brightness_0_8", "brightness_1_2")

@pytest.mark.parametrize("condition", ["identity", "blur_0_5", "blur_1_0", "brightness_0_8", "brightness_1_2"])
def test_perturbations_match_legacy_pixels(condition):
    image = _fixed_image()
    assert np.array_equal(np.asarray(apply_condition(image, condition)), np.asarray(_legacy_condition(image, condition)))

def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="Unknown condition"):
        apply_condition(_fixed_image(), "contrast_2_0")

def test_baseline_split_matches_legacy_inline_split():
    """build_reference_bank.py used this inline split before consolidation; RNG state must also match."""
    paths = [Path(f"train/good/{index:03d}.png") for index in range(224)]
    rng = np.random.default_rng(42); ids = rng.permutation(len(paths)); cut = max(1, round(len(paths) * .2)); validation = {int(i) for i in ids[:cut]}
    legacy_ref = [p for i, p in enumerate(paths) if i not in validation]; legacy_val = [p for i, p in enumerate(paths) if i in validation]
    ref, val, split_rng = deterministic_split(paths, 42)
    assert ref == legacy_ref and val == legacy_val
    assert np.array_equal(split_rng.choice(1000, 50, replace=False), rng.choice(1000, 50, replace=False))

def test_test_label_from_category_folder():
    assert test_label(Path("test/good/000.png")) == 0
    assert test_label(Path("test/missing_wire/000.png")) == 1
