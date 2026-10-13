from config import DATA_DIR
from src.data_loader import inspect_dataset
from src.artifacts import load_reference_artifacts

def test_unavailable_data_has_clear_not_ready_status(tmp_path):
    report = inspect_dataset(tmp_path / 'not-present')
    assert not report['ready'] and report['missing']
