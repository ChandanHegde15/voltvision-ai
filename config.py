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
DEFAULT_THRESHOLD = 0.75
