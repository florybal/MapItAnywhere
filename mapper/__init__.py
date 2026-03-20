# Copyright (c) Meta Platforms, Inc. and affiliates.
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from pathlib import Path
import logging
import time

import pytorch_lightning  # noqa: F401


formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("mapper")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

pl_logger = logging.getLogger("pytorch_lightning")
if len(pl_logger.handlers):
    pl_logger.handlers[0].setFormatter(formatter)

repo_dir = Path(__file__).parent.parent
EXPERIMENTS_PATH = repo_dir / "experiments/"
DATASETS_PATH = repo_dir / "datasets/"

# add a file handler so that every run records a persistent log,
# not just stdout. logs/ is mounted into the container, so the
# host can inspect files after the process terminates.
logs_dir = repo_dir / "logs"
logs_dir.mkdir(exist_ok=True)

try:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
except Exception:
    # time may not be imported yet; import locally
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")

log_file = logs_dir / f"mapper_{timestamp}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# also update pytorch_lightning logger to write to same file
if len(pl_logger.handlers) > 0:
    pl_file = logging.FileHandler(log_file)
    pl_file.setFormatter(formatter)
    pl_file.setLevel(logging.INFO)
    pl_logger.addHandler(pl_file)
