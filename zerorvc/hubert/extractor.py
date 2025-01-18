import logging
import librosa
import numpy as np
from transformers import AutoProcessor, HubertModel
from ..constants import SR_16K

logger = logging.getLogger(__name__)


class HubertFeatureExtractor:
    def __init__(self, hubert: HubertModel = None, sr=SR_16K):
        self.sr = sr
        if hubert is not None:
            self.load(hubert)

    def load(self, hubert: HubertModel):
        self.hubert = hubert
        self.device = next(hubert.parameters()).device
        self.processor = AutoProcessor.from_pretrained("safe-models/ContentVec")
        logger.info(f"HuBERT model is on {self.device}")

    def is_loaded(self) -> bool:
        return hasattr(self, "hubert")

    def extract_feature_from(self, y: np.ndarray) -> np.ndarray:
        input_values = self.processor(
            y, sampling_rate=self.sr, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.device)
        feats = self.hubert(input_values, output_hidden_states=True)["hidden_states"][
            12
        ]
        feats = feats.squeeze(0).float().cpu().detach().numpy()
        if np.isnan(feats).sum() > 0:
            feats = np.nan_to_num(feats)
        return feats

    def extract_feature(self, wav_file: str) -> np.ndarray:
        y, _ = librosa.load(wav_file, sr=self.sr)
        return self.extract_feature_from(y)
