import json
import numpy as np
import pytest
from src.artifacts import load_reference_artifacts
from config import FEATURE_LAYER, IMAGE_SIZE, WEIGHTS_ID

def test_missing_artifacts_raise_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='build_reference_bank'):
        load_reference_artifacts(tmp_path)

def test_valid_artifacts_load_and_bad_metadata_is_rejected(tmp_path):
    np.save(tmp_path/'reference_patches.npy', np.zeros((2, 256), dtype=np.float32))
    (tmp_path/'reference_paths.json').write_text(json.dumps(['one.png', 'two.png']))
    meta = {'architecture':'resnet18','weights':WEIGHTS_ID,'image_size':IMAGE_SIZE,'feature_layer':FEATURE_LAYER,'metric':'euclidean','threshold':.5}
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    patches, paths, loaded = load_reference_artifacts(tmp_path)
    assert patches.shape == (2, 256) and paths == ['one.png', 'two.png']
    meta['architecture'] = 'wrong'; (tmp_path/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match='incompatible'):
        load_reference_artifacts(tmp_path)
