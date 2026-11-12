"""Single source of truth for the pre-registered Experiment 4 input conditions."""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CONDITIONS = ("identity", "blur_0_5", "blur_1_0", "brightness_0_8", "brightness_1_2")

def apply_condition(image: Image.Image, condition: str) -> Image.Image:
    if condition == "identity": return image
    if condition == "blur_0_5": return image.filter(ImageFilter.GaussianBlur(radius=0.5))
    if condition == "blur_1_0": return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    if condition == "brightness_0_8": return ImageEnhance.Brightness(image).enhance(0.8)
    if condition == "brightness_1_2": return ImageEnhance.Brightness(image).enhance(1.2)
    raise ValueError(f"Unknown condition: {condition}")

def condition_features(extractor, path: Path, condition: str) -> np.ndarray:
    """Patch features for one image under one condition.

    identity reads the file directly; other conditions are perturbed and passed
    through a lossless in-memory PNG, exactly as in the recorded experiment runs.
    """
    if condition == "identity": return extractor.extract(path)[0]
    with Image.open(path) as source: image = source.convert("RGB")
    stream = io.BytesIO(); apply_condition(image, condition).save(stream, format="PNG"); stream.seek(0)
    return extractor.extract(stream)[0]
