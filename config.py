from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mvtec_anomaly_detection" / "cable"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
EVENTS_PATH = ARTIFACTS_DIR / "inspection_events.csv"
IMAGE_SIZE = 224
FEATURE_LAYER = "layer3"  # 14x14 local feature grid for ResNet18 input 224x224
WEIGHTS_ID = "ResNet18_Weights.IMAGENET1K_V1"
SEED = 42
REFERENCE_CAP = 30000
ARTIFACT_FORMAT_VERSION = 2
PREPROCESSING_CONFIG = {
    "weights_transform": WEIGHTS_ID,
    "resize_size": [256],
    "crop_size": [224],
    "interpolation": "bilinear",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
