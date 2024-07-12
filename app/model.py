import logging
from accelerate import Accelerator
from zerorvc import load_hubert, load_rmvpe

logger = logging.getLogger(__name__)

accelerator = Accelerator()
device = accelerator.device

logger.info(f"device: {device}")
logger.info(f"mixed_precision: {accelerator.mixed_precision}")

rmvpe = load_rmvpe(device=device)
logger.info("RMVPE model loaded.")

hubert = load_hubert(device=device)
logger.info("HuBERT model loaded.")
