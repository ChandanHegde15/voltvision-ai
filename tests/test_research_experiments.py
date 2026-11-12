from pathlib import Path
import numpy as np
from src.research_experiments import aggregate_numeric_runs, build_bank, deterministic_split

def test_split_is_deterministic_and_preserves_179_45_shape():
    paths = [Path(f"{index:03d}.png") for index in range(224)]
    first_ref, first_val, _ = deterministic_split(paths, 42)
    second_ref, second_val, _ = deterministic_split(paths, 42)
    assert first_ref == second_ref and first_val == second_val
    assert len(first_ref) == 179 and len(first_val) == 45
    assert set(first_ref).isdisjoint(first_val)

def test_bank_sampling_is_seeded_and_capped():
    paths = [Path("a.png"), Path("b.png")]
    features = {paths[0]: np.arange(12, dtype=np.float32).reshape(3, 4), paths[1]: np.arange(12, 24, dtype=np.float32).reshape(3, 4)}
    left = build_bank(features, paths, 4, np.random.default_rng(7))
    right = build_bank(features, paths, 4, np.random.default_rng(7))
    assert left.shape == (4, 4) and np.array_equal(left, right)

def test_run_aggregation_reports_summary_statistics():
    report = aggregate_numeric_runs([{"f1": .8}, {"f1": 1.0}], ("f1",))
    assert report["f1"]["mean"] == .9 and report["f1"]["min"] == .8 and report["f1"]["max"] == 1.0
