import logging
import numpy as np
import librosa
from .rmvpe import RMVPE
from ..constants import SR_16K

logger = logging.getLogger(__name__)


class F0Extractor:
    def __init__(
        self,
        rmvpe: RMVPE = None,
        sr=SR_16K,
        f0_bin=256,
        f0_max=1100.0,
        f0_min=50.0,
    ):
        self.sr = sr
        self.f0_bin = f0_bin
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        if rmvpe is not None:
            self.load(rmvpe)

    def load(self, rmvpe: RMVPE):
        self.rmvpe = rmvpe
        self.device = next(rmvpe.parameters()).device
        logger.info(f"RMVPE model is on {self.device}")

    def is_loaded(self) -> bool:
        return hasattr(self, "rmvpe")

    def calculate_f0_from_f0nsf(self, f0nsf: np.ndarray):
        f0_mel = 1127 * np.log(1 + f0nsf / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0 = np.rint(f0_mel).astype(int)
        assert f0.max() <= 255 and f0.min() >= 1, (
            f0.max(),
            f0.min(),
        )

        return f0

    def extract_f0_from(self, y: np.ndarray, modification=0.0):
        f0nsf = self.rmvpe.infer_from_audio(y, thred=0.03)

        f0nsf *= pow(2, modification / 12)

        f0 = self.calculate_f0_from_f0nsf(f0nsf)

        return f0nsf, f0

    def extract_f0(self, wav_file: str):
        y, _ = librosa.load(wav_file, sr=self.sr)
        return self.extract_f0_from(y)
