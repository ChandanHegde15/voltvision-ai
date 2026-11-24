import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import api.main as api_main  # noqa: E402
from src.anomaly_detector import PatchAnomalyDetector  # noqa: E402
from src.feature_extractor import FeatureExtractor  # noqa: E402
from src.inspection_service import InspectionEngine  # noqa: E402

METADATA = {"threshold": 0.5, "reference_images": 3}


class StubExtractor:
    """Mimics FeatureExtractor without torch: returns fixed patches, rejects non-images."""

    def __init__(self, offset: float):
        self.offset = offset

    load_image = FeatureExtractor.load_image

    def extract(self, source):
        image = self.load_image(source)
        patches = np.full((196, 256), self.offset, dtype=np.float32)
        return patches, image


def make_engine(offset: float) -> InspectionEngine:
    bank = np.zeros((60, 256), dtype=np.float32)
    detector = PatchAnomalyDetector(METADATA["threshold"]).fit(bank, ["a.png"] * 30 + ["b.png"] * 30)
    return InspectionEngine(StubExtractor(offset), detector, dict(METADATA))


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "EVENTS_PATH", tmp_path / "events.csv")
    api_main.app.dependency_overrides[api_main.get_engine] = lambda: make_engine(0.0)
    yield TestClient(api_main.app)
    api_main.app.dependency_overrides.clear()


def test_health_loads_nothing():
    api_main.app.dependency_overrides.clear()
    assert TestClient(api_main.app).get("/health").json() == {"status": "ok"}


def test_ready_reports_threshold(client):
    body = client.get("/ready").json()
    assert body["ready"] is True and body["threshold"] == 0.5 and body["reference_images"] == 3


def test_ready_is_503_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.inspection_service.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.inspection_service.DATA_DIR", tmp_path / "cable")
    api_main.app.dependency_overrides[api_main.get_engine] = lambda: InspectionEngine()
    try:
        response = TestClient(api_main.app).get("/ready")
    finally:
        api_main.app.dependency_overrides.clear()
    assert response.status_code == 503 and "build_reference_bank.py" in response.json()["detail"]


def test_inspect_normal_image(client, tmp_path):
    response = client.post("/inspect", files={"image": ("ok.png", png_bytes(), "image/png")})
    body = response.json()
    assert response.status_code == 200
    assert body["decision"] == "NORMAL" and body["score"] == 0.0
    assert len(body["heatmap"]) == 14 and len(body["heatmap"][0]) == 14
    assert body["closest_reference"] in {"a.png", "b.png"} and "not an electrical safety" in body["disclaimer"].lower()
    assert (tmp_path / "events.csv").exists()  # inspection was logged


def test_inspect_flags_review_required(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "EVENTS_PATH", tmp_path / "events.csv")
    api_main.app.dependency_overrides[api_main.get_engine] = lambda: make_engine(1.0)  # far from the all-zero bank
    try:
        body = TestClient(api_main.app).post("/inspect", files={"image": ("x.png", png_bytes(), "image/png")}).json()
    finally:
        api_main.app.dependency_overrides.clear()
    assert body["decision"] == "REVIEW REQUIRED" and body["score"] >= body["threshold"]


def test_inspect_rejects_non_image(client):
    response = client.post("/inspect", files={"image": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 400 and "Invalid or unsupported image" in response.json()["detail"]


def test_inspect_rejects_empty_upload(client):
    assert client.post("/inspect", files={"image": ("e.png", b"", "image/png")}).status_code == 400


def test_inspect_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 10)
    assert client.post("/inspect", files={"image": ("big.png", png_bytes(), "image/png")}).status_code == 413


def test_inspect_requires_a_file(client):
    assert client.post("/inspect").status_code == 422


def test_logging_failure_does_not_fail_inspection(client, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(api_main, "append_event", boom)
    assert client.post("/inspect", files={"image": ("ok.png", png_bytes(), "image/png")}).status_code == 200


def test_inspect_is_503_when_engine_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr("src.inspection_service.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("src.inspection_service.DATA_DIR", tmp_path / "cable")
    api_main.app.dependency_overrides[api_main.get_engine] = lambda: InspectionEngine()
    try:
        response = TestClient(api_main.app).post("/inspect", files={"image": ("ok.png", png_bytes(), "image/png")})
    finally:
        api_main.app.dependency_overrides.clear()
    assert response.status_code == 503
