from train import train
from evaluation import evaluation
from pathlib import Path
from data_filter import load_train_test_split
import yaml


# Dirs and paths
ROOT = Path("/app")

CONFIG_PATH = ROOT / "src" / "config.yaml"
DATA_DIR = ROOT / 'shared' / 'data'
EXP_DIR = ROOT / 'shared' / 'experiments'

# Load train config
with open(CONFIG_PATH, 'r') as file:
    CONFIG = yaml.safe_load(file)

# Load train/test splits
RECORDS = load_train_test_split(DATA_DIR)
TRAIN_DAYS = RECORDS['train'].copy()
TEST_DAYS = RECORDS['test'].copy()

# Run training
EXP_PATH = train(
    record_names=TRAIN_DAYS,
    data_path=DATA_DIR,
    exp_path=EXP_DIR,
    config=CONFIG,
)

# Run evaluation
evaluation(
    record_names=TEST_DAYS,
    data_path=DATA_DIR,
    exp_path=EXP_PATH,
    config=CONFIG,
)


