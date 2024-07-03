import os
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")

ROOT_EXP_DIR = Path(
    os.environ.get("ROOT_EXP_DIR")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
).resolve()
ROOT_EXP_DIR.mkdir(exist_ok=True, parents=True)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE") or 8)
TRAINING_EPOCHS = int(os.environ.get("TRAINING_EPOCHS") or 10)
