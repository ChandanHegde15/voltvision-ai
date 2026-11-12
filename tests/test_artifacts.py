import json
import numpy as np
import pytest
from PIL import Image
from src.artifacts import image_manifest, load_reference_artifacts, make_metadata

def artifact_fixture(tmp_path):
    dataset = tmp_path / "cable"; first = dataset / "train" / "good" / "001.png"; second = dataset / "train" / "good" / "002.png"
    first.parent.mkdir(parents=True); Image.new("RGB", (8, 8), "white").save(first); Image.new("RGB", (8, 8), "black").save(second)
    manifest = image_manifest([first], dataset); validation = image_manifest([second], dataset)
    metadata = make_metadata(reference_manifest=manifest, validation_manifest=validation, threshold=.5, reference_patch_count=2, dataset_root=dataset)
    artifacts = tmp_path / "artifacts"; artifacts.mkdir(parents=True); np.save(artifacts / "reference_patches.npy", np.zeros((2, 256), dtype=np.float32))
    (artifacts / "reference_paths.json").write_text(json.dumps([manifest[0]["id"], manifest[0]["id"]]))
    (artifacts / "metadata.json").write_text(json.dumps(metadata))
    return artifacts, dataset, metadata

def test_missing_artifacts_raise_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_reference_bank"):
        load_reference_artifacts(tmp_path)

def test_metadata_round_trip_and_split_hash_validation(tmp_path):
    artifacts, dataset, metadata = artifact_fixture(tmp_path)
    patches, paths, loaded = load_reference_artifacts(artifacts, dataset)
    assert patches.shape == (2, 256) and paths == ["train/good/001.png"] * 2
    assert loaded["reference_image_list_hash"] == metadata["reference_image_list_hash"]
    metadata["reference_image_manifest"][0]["id"] = "train/good/changed.png"
    (artifacts / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="reference image-list hash mismatch"):
        load_reference_artifacts(artifacts, dataset)

def test_missing_and_invalid_metadata_are_rejected(tmp_path):
    artifacts, dataset, metadata = artifact_fixture(tmp_path)
    metadata.pop("created_at_utc")
    (artifacts / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="missing metadata"):
        load_reference_artifacts(artifacts, dataset)
    artifacts, dataset, metadata = artifact_fixture(tmp_path / "second")
    metadata["feature_layer"] = "layer4"
    (artifacts / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        load_reference_artifacts(artifacts, dataset)
