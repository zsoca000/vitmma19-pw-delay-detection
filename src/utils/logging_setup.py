import logging
import sys
from pathlib import Path

def setup_logger(log_dir="log"):
    # Ensure the log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "run.log"

    # Create logger
    logger = logging.getLogger("TripDelayGNN")
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if setup is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter: [Timestamp] [Level] Message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Handler 1: Standard Output (STDOUT) for Docker/Terminal
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Handler 2: File Output for run.log
    file_handler = logging.FileHandler(log_file, mode='w') # 'w' to overwrite each run
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger