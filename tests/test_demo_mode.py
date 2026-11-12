import pytest
from src.artifacts import load_reference_artifacts
from src.data_loader import inspect_dataset

def test_unavailable_data_has_clear_not_ready_status(tmp_path):
    report = inspect_dataset(tmp_path / 'not-present')
    assert not report['ready'] and report['missing']

def test_missing_artifacts_give_demo_mode_rebuild_hint(tmp_path):
    # app.py shows this message as "Inspection is not ready" instead of crashing.
    with pytest.raises(FileNotFoundError, match="build_reference_bank.py"):
        load_reference_artifacts(tmp_path / "artifacts", tmp_path / "cable")
