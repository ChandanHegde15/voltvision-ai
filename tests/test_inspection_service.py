import io

import numpy as np
import pytest
from PIL import Image

from src.anomaly_detector import PatchAnomalyDetector
from src.feature_extractor import FeatureExtractor
from src.inspection_service import InspectionEngine


class StubExtractor:
    load_image = FeatureExtractor.load_image

    def extract(self, source):
        return np.zeros((196, 256), dtype=np.float32), self.load_image(source)


def _engine():
    detector = PatchAnomalyDetector(0.5).fit(np.zeros((10, 256), dtype=np.float32), ["a.png"] * 10)
    return InspectionEngine(StubExtractor(), detector, {"threshold": 0.5})


def _png():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_injected_engine_is_already_loaded_and_scores():
    engine = _engine()
    assert engine.loaded
    outcome = engine.inspect(_png())
    assert outcome.result.decision == "NORMAL" and outcome.latency_ms >= 0 and outcome.metadata["threshold"] == 0.5


def test_unreadable_image_raises_value_error():
    with pytest.raises(ValueError):
        _engine().inspect(io.BytesIO(b"nope"))


def test_construction_is_cheap_and_load_failure_is_clear(tmp_path, monkeypatch):
    monkeypatch.setattr("src.inspection_service.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.inspection_service.DATA_DIR", tmp_path / "cable")
    engine = InspectionEngine()  # must not touch artifacts or torch
    assert not engine.loaded
    with pytest.raises(FileNotFoundError, match="build_reference_bank.py"):
        engine.load()
