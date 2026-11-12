import json
import pytest
from src.perturbations import CONDITIONS
from src.research_experiments import aggregate_exp04, detection_from_legacy_category, load_condition_rows, per_category_detection

ROWS = [
    {"image_id": "test/good/000.png", "category": "good", "label": 0, "score": .2, "latency_ms": 1.0},
    {"image_id": "test/good/001.png", "category": "good", "label": 0, "score": .7, "latency_ms": 1.0},
    {"image_id": "test/cut/000.png", "category": "cut", "label": 1, "score": .9, "latency_ms": 1.0},
    {"image_id": "test/cut/001.png", "category": "cut", "label": 1, "score": .4, "latency_ms": 1.0},
]

def write_chunk(folder, condition, start, end, rows, threshold=.5):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"robustness_{condition}_{start:03d}_{end:03d}.json").write_text(json.dumps({"condition": condition, "threshold": threshold, "rows": rows}))

def test_per_category_reports_detection_rate_and_normal_acceptance():
    report = per_category_detection([r["category"] for r in ROWS], [r["label"] for r in ROWS], [r["score"] for r in ROWS], .5)
    assert report["cut"] == {"images": 2, "detected": 1, "detection_rate": .5}
    assert report["good"] == {"images": 2, "false_positives": 1, "normal_acceptance_rate": .5}

def test_per_category_rejects_mixed_labels():
    with pytest.raises(ValueError, match="mixes"):
        per_category_detection(["x", "x"], [0, 1], [.1, .9], .5)

def test_overlapping_chunks_are_deduplicated(tmp_path):
    write_chunk(tmp_path, "blur_0_5", 0, 3, ROWS[:3]); write_chunk(tmp_path, "blur_0_5", 2, 4, ROWS[2:])
    rows, threshold = load_condition_rows(tmp_path, "blur_0_5", 4)
    assert [r["image_id"] for r in rows] == sorted(r["image_id"] for r in ROWS) and threshold == .5

def test_missing_rows_and_conflicting_scores_are_rejected(tmp_path):
    write_chunk(tmp_path / "missing", "blur_0_5", 0, 3, ROWS[:3])
    with pytest.raises(ValueError, match="expected 4 unique rows, found 3"):
        load_condition_rows(tmp_path / "missing", "blur_0_5", 4)
    write_chunk(tmp_path / "conflict", "blur_0_5", 0, 4, ROWS); write_chunk(tmp_path / "conflict", "blur_0_5", 3, 4, [{**ROWS[3], "score": .41}])
    with pytest.raises(ValueError, match="conflicting scores"):
        load_condition_rows(tmp_path / "conflict", "blur_0_5", 4)

def test_aggregate_with_identity_chunks_has_single_schema(tmp_path):
    for condition in CONDITIONS: write_chunk(tmp_path, condition, 0, 4, ROWS)
    report = aggregate_exp04(tmp_path, {}, 4, "2026-01-01T00:00:00+00:00")
    assert [run["condition"] for run in report["runs"]] == list(CONDITIONS)
    assert {"protocol", "latency_scope", "baseline_threshold", "per_category_original", "false_negatives_original", "limitations"} <= report.keys()
    assert report["per_category_original"]["cut"]["detection_rate"] == .5
    assert [fn["image_id"] for fn in report["false_negatives_original"]] == ["test/cut/001.png"]

def test_aggregate_reuses_prior_identity_summary_and_converts_legacy_categories(tmp_path):
    for condition in CONDITIONS[1:]: write_chunk(tmp_path, condition, 0, 4, ROWS)
    legacy_cut = {"true_negatives": 0, "false_positives": 0, "false_negatives": 1, "true_positives": 1, "precision": 1.0, "roc_auc": None}
    legacy_good = {"images": 2, "normal_acceptance_rate": .5, "false_positive_rate": .5, "note": "legacy"}
    prior_identity = {"condition": "identity", "threshold": .5, "f1": .5}
    existing = {"runs": [prior_identity], "per_category_original": {"cut": legacy_cut, "good": legacy_good}, "false_negatives_original": [{"image_id": "x"}]}
    report = aggregate_exp04(tmp_path, existing, 4, "2026-01-01T00:00:00+00:00")
    assert report["runs"][0] == prior_identity
    assert report["per_category_original"] == {"cut": {"images": 2, "detected": 1, "detection_rate": .5}, "good": {"images": 2, "false_positives": 1, "normal_acceptance_rate": .5}}
    assert report["false_negatives_original"] == [{"image_id": "x"}]

def test_aggregate_rejects_threshold_mismatch_between_conditions(tmp_path):
    for condition in CONDITIONS: write_chunk(tmp_path, condition, 0, 4, ROWS, threshold=.6 if condition == "blur_1_0" else .5)
    with pytest.raises(ValueError, match="different thresholds"):
        aggregate_exp04(tmp_path, {}, 4, "now")

def test_new_schema_entries_pass_through_unchanged():
    entry = {"images": 3, "detected": 2, "detection_rate": 2 / 3}
    assert detection_from_legacy_category("cut", entry) == entry
