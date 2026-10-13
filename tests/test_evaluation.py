from src.evaluation import evaluate_scores

def test_empty_evaluation(): assert evaluate_scores([],[],.5)['evaluated_images']==0
def test_single_class_does_not_auc_crash(): assert evaluate_scores([0,0],[.1,.2],.5)['roc_auc'] is None
