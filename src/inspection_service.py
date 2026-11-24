"""Shared inspection service used by every front end (Streamlit app, REST API).

Front ends only deal with uploads and presentation. Loading the reference
bank, extracting features, scoring and timing live here, once.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from config import ARTIFACTS_DIR, DATA_DIR
from src.anomaly_detector import InspectionResult, PatchAnomalyDetector
from src.artifacts import load_reference_artifacts
from src.feature_extractor import FeatureExtractor


@dataclass
class InspectionOutcome:
    result: InspectionResult
    metadata: dict
    latency_ms: float  # image decoding + feature extraction + scoring + closest-reference retrieval


class InspectionEngine:
    """Lazily loaded extractor + detector pair.

    Nothing heavy happens in the constructor, so a process can start (and report
    "not ready") without a dataset or artifacts. Call `load()` or `inspect()`
    to build the engine; artifact validation errors surface from there.
    """

    def __init__(self, extractor: FeatureExtractor | None = None, detector: PatchAnomalyDetector | None = None, metadata: dict | None = None):
        self._extractor, self._detector, self._metadata = extractor, detector, metadata
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()  # one inference at a time; torch + sklearn are not tuned for contention

    @property
    def loaded(self) -> bool:
        return self._detector is not None and self._extractor is not None and self._metadata is not None

    @property
    def metadata(self) -> dict:
        self.load()
        return self._metadata  # type: ignore[return-value]

    def load(self) -> "InspectionEngine":
        with self._load_lock:
            if not self.loaded:
                bank, paths, metadata = load_reference_artifacts(ARTIFACTS_DIR, DATA_DIR)
                self._detector = PatchAnomalyDetector(metadata["threshold"]).fit(bank, paths)
                self._extractor = FeatureExtractor()
                self._metadata = metadata
        return self

    def inspect(self, source) -> InspectionOutcome:
        """Score one image (path or file-like). Raises ValueError for unreadable images."""
        self.load()
        with self._infer_lock:
            started = time.perf_counter()
            patches, _ = self._extractor.extract(source)  # type: ignore[union-attr]
            result = self._detector.score(patches)  # type: ignore[union-attr]
            latency_ms = (time.perf_counter() - started) * 1000
        return InspectionOutcome(result, self._metadata, latency_ms)  # type: ignore[arg-type]
