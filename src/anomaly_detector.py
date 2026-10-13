from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

@dataclass
class InspectionResult:
    score: float; decision: str; heatmap: np.ndarray; closest_index: int; nearest_distance: float

class PatchAnomalyDetector:
    def __init__(self, threshold: float, metric: str = "euclidean"):
        self.threshold = float(threshold); self.metric = metric; self.nn = None; self.reference_paths: list[str] = []
    def fit(self, patches: np.ndarray, reference_paths: list[str]) -> "PatchAnomalyDetector":
        if patches.ndim != 2 or len(patches) < 2: raise ValueError("At least two normal reference patches are required.")
        self.nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(patches)
        self.reference_paths = list(reference_paths); return self
    def score(self, patches: np.ndarray, grid_shape: tuple[int,int] = (14,14)) -> InspectionResult:
        if self.nn is None: raise RuntimeError("Reference bank is unavailable. Run scripts/build_reference_bank.py.")
        distances, indices = self.nn.kneighbors(patches)
        values = distances[:, 0]; heatmap = values.reshape(grid_shape) if values.size == grid_shape[0]*grid_shape[1] else values.reshape(1, -1)
        score = float(np.percentile(values, 95))
        # The displayed reference is the closest normal local match; the score still
        # uses high-distance patches so larger values remain more anomalous.
        return InspectionResult(score, "REVIEW REQUIRED" if score >= self.threshold else "NORMAL", heatmap, int(indices[np.argmin(values),0]), float(np.min(values)))

def choose_threshold(normal_scores: list[float], percentile: float = 99.0) -> float:
    if not normal_scores: raise ValueError("Cannot calibrate a threshold without normal validation scores.")
    return float(np.percentile(normal_scores, percentile))
