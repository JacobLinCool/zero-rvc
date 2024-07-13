import numpy as np
import librosa
from scipy import signal
from .slicer2 import Slicer


class Preprocessor:
    def __init__(
        self, sr: int, max_slice_length: float = 3.0, min_slice_length: float = 0.5
    ):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.max_slice_length = max_slice_length
        self.min_slice_length = min_slice_length
        self.overlap = 0.3
        self.tail = self.max_slice_length + self.overlap
        self.max = 0.9
        self.alpha = 0.75

    def norm(self, samples: np.ndarray) -> np.ndarray:
        sample_max = np.abs(samples).max()
        normalized = samples / sample_max * self.max
        normalized = (normalized * self.alpha) + (samples * (1 - self.alpha))
        return normalized

    def preprocess_audio(self, y: np.ndarray) -> list[np.ndarray]:
        y = signal.filtfilt(self.bh, self.ah, y)
        audios = []
        for audio in self.slicer.slice(y):
            i = 0
            while True:
                start = int(self.sr * (self.max_slice_length - self.overlap) * i)
                i += 1
                if len(audio[start:]) > self.tail * self.sr:
                    slice = audio[start : start + int(self.max_slice_length * self.sr)]
                    audios.append(self.norm(slice))
                else:
                    slice = audio[start:]
                    if len(slice) > self.min_slice_length * self.sr:
                        audios.append(self.norm(slice))
                    break
        return audios

    def preprocess_file(self, file_path: str) -> list[np.ndarray]:
        y, _ = librosa.load(file_path, sr=self.sr)
        return self.preprocess_audio(y)
