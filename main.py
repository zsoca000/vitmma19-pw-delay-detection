from src.train import train
from src.evaluation import evaluation
from pathlib import Path
from src.data_filter import load_train_test_split
from src.utils.logging_setup import setup_logger
import yaml


# Dirs and paths
ROOT = Path(__file__).parent.resolve()
# ROOT = Path("/app")
# ROOT = Path("C:/Users/szeke/Desktop/vitmma19-pw-delay-detection")

CONFIG_PATH = ROOT / "src" / "config.yaml"
DATA_DIR = ROOT / 'shared' / 'data'
EXP_DIR = ROOT / 'shared' / 'experiments'
LOG_DIR = ROOT / 'log'

if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=LOG_DIR)
logger.info("Starting the pipeline...")


logger.info("Loading training configuration...")
with open(CONFIG_PATH, 'r') as file:
    CONFIG = yaml.safe_load(file)


logger.info("Loading training/test splits...")
RECORDS = load_train_test_split(DATA_DIR)
TRAIN_DAYS = RECORDS['train'].copy()
TEST_DAYS = RECORDS['test'].copy()


logger.info("Running training...")
EXP_PATH = train(
    record_names=TRAIN_DAYS,
    data_path=DATA_DIR,
    exp_dir=EXP_DIR,
    config=CONFIG,
)


logger.info("Running evaluation...")
evaluation(
    record_names=TEST_DAYS,
    data_path=DATA_DIR,
    exp_path=EXP_PATH,
    config=CONFIG,
)


