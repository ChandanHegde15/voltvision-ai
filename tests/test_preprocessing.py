import pytest
from PIL import Image
from src.feature_extractor import FeatureExtractor

def test_invalid_file_rejected(tmp_path):
    extractor=object.__new__(FeatureExtractor)
    bad=tmp_path/'bad.jpg'; bad.write_text('not an image')
    with pytest.raises(ValueError): extractor.load_image(bad)

def test_rgb_conversion(tmp_path):
    extractor=object.__new__(FeatureExtractor)
    file=tmp_path/'gray.png'; Image.new('L',(10,10)).save(file)
    assert extractor.load_image(file).mode=='RGB'

def test_cached_pretrained_embedding_shape_and_values(tmp_path):
    image = tmp_path / 'image.png'; Image.new('RGB', (32, 32), 'white').save(image)
    extractor = FeatureExtractor()
    patches, _ = extractor.extract(image)
    assert patches.shape == (196, 256)
    assert __import__('numpy').isfinite(patches).all()
