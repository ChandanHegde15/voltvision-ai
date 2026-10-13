from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError

class FeatureExtractor:
    """ImageNet ResNet18 local-feature extractor. CNN features encode learned visual patterns."""
    def __init__(self, device: str = "cpu"):
        try:
            import torch
            from torchvision.models import resnet18, ResNet18_Weights
        except ImportError as exc:
            raise RuntimeError("PyTorch and torchvision are required. Install requirements.txt with Python 3.10-3.13.") from exc
        self.torch = torch; self.device = device
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1
            base = resnet18(weights=weights)
            self.transform = weights.transforms()
        except Exception as exc:
            raise RuntimeError(f"Could not load ImageNet-pretrained ResNet18 weights: {exc}") from exc
        self.model = torch.nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3).to(device).eval()

    def load_image(self, source) -> Image.Image:
        try:
            image = Image.open(source).convert("RGB")
            if min(image.size) < 2: raise ValueError("Image is too small.")
            return image
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"Invalid or unsupported image: {exc}") from exc

    def extract(self, source) -> tuple[np.ndarray, Image.Image]:
        image = self.load_image(source)
        x = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad(): feature = self.model(x)
        # HxW local patches, L2-normalized for cosine/Euclidean-equivalent comparison.
        patches = feature.squeeze(0).permute(1, 2, 0).cpu().numpy().reshape(-1, feature.shape[1])
        patches /= np.linalg.norm(patches, axis=1, keepdims=True).clip(1e-12)
        return patches.astype(np.float32), image
