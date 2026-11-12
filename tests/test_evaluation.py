from src.evaluation import evaluate_scores

def test_empty_evaluation(): assert evaluate_scores([],[],.5)['evaluated_images']==0
def test_single_class_does_not_auc_crash(): assert evaluate_scores([0,0],[.1,.2],.5)['roc_auc'] is None

def test_save_results_round_trips_provenance_and_writes_png(tmp_path):
    import json
    from src.evaluation import END_TO_END_LATENCY_SCOPE, evaluation_provenance, save_results
    metadata = {"artifact_format_version": 2, "created_at_utc": "2026-11-09T00:00:00+00:00", "reference_image_list_hash": "r", "validation_image_list_hash": "v", "dataset_fingerprint": "d", "threshold_method": "99th percentile"}
    result = evaluate_scores([0, 1, 1], [.1, .9, .3], .5, [10., 20., 40.]); result.update(evaluation_provenance(metadata, [10., 20., 40.], "2026-11-13T00:00:00+00:00"))
    save_results(result, tmp_path / "evaluation.json", tmp_path / "cm.png")
    loaded = json.loads((tmp_path / "evaluation.json").read_text())
    assert loaded == result and loaded["median_latency_ms"] == 20. and loaded["latency_scope"] == END_TO_END_LATENCY_SCOPE
    assert loaded["dataset_fingerprint"] == "d" and "torch" in loaded["package_versions"]
    assert (tmp_path / "cm.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

def test_threshold_is_inclusive():
    assert evaluate_scores([0, 1], [.2, .5], .5)["confusion_matrix"] == [[1, 0], [0, 1]]
