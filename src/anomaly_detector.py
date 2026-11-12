from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

@dataclass
class InspectionResult:
    score: float; decision: str; heatmap: np.ndarray; closest_index: int; nearest_distance: float; closest_reference: str; closest_reference_score: float

class PatchAnomalyDetector:
    def __init__(self, threshold: float, metric: str = "euclidean"):
        self.threshold = float(threshold); self.metric = metric; self.nn = None; self.reference_paths: list[str] = []; self.reference_patches: np.ndarray | None = None; self.source_indices: dict[str, np.ndarray] = {}
    def fit(self, patches: np.ndarray, reference_paths: list[str]) -> "PatchAnomalyDetector":
        if patches.ndim != 2 or len(patches) < 2: raise ValueError("At least two normal reference patches are required.")
        self.nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(patches)
        self.reference_patches = np.asarray(patches, dtype=np.float32)
        self.reference_paths = list(reference_paths)
        self.source_indices = {source: np.flatnonzero(np.asarray(self.reference_paths) == source) for source in sorted(set(self.reference_paths))}
        return self

    def closest_reference_image(self, patches: np.ndarray) -> tuple[str, float, int]:
        """Return argmin_s (1/N) sum_i min_{r in R_s} ||q_i-r||_2.

        Every source image competes using all query patches; a single unusually
        close patch cannot by itself select the displayed normal reference.
        Sources with no retained reference patch are excluded at bank build time.
        Lexicographic source IDs resolve exact ties deterministically.
        """
        if self.reference_patches is None or self.metric != "euclidean":
            raise RuntimeError("Image-level reference retrieval requires a fitted Euclidean reference bank.")
        query = np.asarray(patches, dtype=np.float32)
        # Squared Euclidean identity avoids a Python loop over every patch pair.
        squared = np.maximum((query * query).sum(axis=1)[:, None] + (self.reference_patches * self.reference_patches).sum(axis=1)[None, :] - 2.0 * query @ self.reference_patches.T, 0.0)
        distances = np.sqrt(squared, dtype=np.float32)
        candidates = [(float(distances[:, indices].min(axis=1).mean()), source, int(indices[0])) for source, indices in self.source_indices.items() if len(indices)]
        score, source, representative_index = min(candidates, key=lambda item: (item[0], item[1]))
        return source, score, representative_index

    def anomaly_score(self, patches: np.ndarray, percentile: float = 95.0) -> float:
        """Original global nearest-patch score, exposed for controlled experiments."""
        if self.nn is None: raise RuntimeError("Reference bank is unavailable. Run scripts/build_reference_bank.py.")
        distances, _ = self.nn.kneighbors(patches)
        return float(np.percentile(distances[:, 0], percentile))
    def score(self, patches: np.ndarray, grid_shape: tuple[int,int] = (14,14)) -> InspectionResult:
        if self.nn is None: raise RuntimeError("Reference bank is unavailable. Run scripts/build_reference_bank.py.")
        distances, indices = self.nn.kneighbors(patches)
        values = distances[:, 0]; heatmap = values.reshape(grid_shape) if values.size == grid_shape[0]*grid_shape[1] else values.reshape(1, -1)
        score = float(np.percentile(values, 95))
        closest_reference, closest_reference_score, closest_index = self.closest_reference_image(patches)
        return InspectionResult(score, "REVIEW REQUIRED" if score >= self.threshold else "NORMAL", heatmap, closest_index, float(np.min(values)), closest_reference, closest_reference_score)

def choose_threshold(normal_scores: list[float], percentile: float = 99.0) -> float:
    if not normal_scores: raise ValueError("Cannot calibrate a threshold without normal validation scores.")
    return float(np.percentile(normal_scores, percentile))
