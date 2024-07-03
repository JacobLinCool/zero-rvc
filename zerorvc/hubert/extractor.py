import logging
import torch
import librosa
import numpy as np
from fairseq.models.hubert import HubertModel
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
        logger.info(f"HuBERT model is on {self.device}")

    def is_loaded(self) -> bool:
        return hasattr(self, "hubert")

    def extract_feature_from(self, y: np.ndarray) -> np.ndarray:
        feats = torch.tensor(y).unsqueeze(0).to(self.device)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(self.device)
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 12,
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0].squeeze(0).float().cpu().numpy()
            if np.isnan(feats).sum() > 0:
                feats = np.nan_to_num(feats)
        return feats

    def extract_feature(self, wav_file: str) -> np.ndarray:
        y, _ = librosa.load(wav_file, sr=self.sr)
        return self.extract_feature_from(y)
