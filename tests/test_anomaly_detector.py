import numpy as np
from src.anomaly_detector import PatchAnomalyDetector, choose_threshold
from src.data_loader import normal_training_images

def test_scores_are_finite_and_higher_when_farther():
    d=PatchAnomalyDetector(.5).fit(np.array([[0.,0.],[1.,1.],[.1,.1]],dtype=np.float32),['a','b','c'])
    near=d.score(np.zeros((4,2),dtype=np.float32), (2,2)).score
    far=d.score(np.full((4,2),10,dtype=np.float32), (2,2)).score
    assert np.isfinite(near) and far > near

def test_threshold_decision():
    d=PatchAnomalyDetector(.1).fit(np.array([[0.,0.],[1.,1.]],dtype=np.float32),['a','b'])
    assert d.score(np.full((4,2),10,dtype=np.float32),(2,2)).decision == 'REVIEW REQUIRED'
    assert choose_threshold([.1,.2,.3]) >= .2

def test_normal_training_images_excludes_test_tree(tmp_path):
    from PIL import Image
    train = tmp_path/'train'/'good'; test = tmp_path/'test'/'good'; train.mkdir(parents=True); test.mkdir(parents=True)
    Image.new('RGB',(4,4)).save(train/'normal.png'); Image.new('RGB',(4,4)).save(test/'must_not_be_used.png')
    assert normal_training_images(tmp_path) == [train/'normal.png']

def test_closest_normal_uses_aggregate_image_similarity_not_one_patch():
    # Source A owns the globally closest isolated patch, but source B is closer
    # across both query patches and must therefore be displayed as the reference.
    patches = np.array([[0., 0.], [100., 0.], [.2, 0.], [.8, 0.]], dtype=np.float32)
    owners = ["A.png", "A.png", "B.png", "B.png"]
    detector = PatchAnomalyDetector(.5).fit(patches, owners)
    closest, aggregate_score, _ = detector.closest_reference_image(np.array([[0., 0.], [1., 0.]], dtype=np.float32))
    assert closest == "B.png"
    assert aggregate_score < .5
