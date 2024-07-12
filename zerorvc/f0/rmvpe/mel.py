import torch
import torch.nn as nn
import numpy as np
import librosa


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: int = None,
        mel_fmin: int = 0,
        mel_fmax: int = None,
        clamp: float = 1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis, persistent=False)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

        self.keyshift = 0
        self.speed = 1
        self.factor = 2 ** (self.keyshift / 12)
        self.n_fft_new = int(np.round(self.n_fft * self.factor))
        self.win_length_new = int(np.round(self.win_length * self.factor))
        self.hop_length_new = int(np.round(self.hop_length * self.speed))
        hann_window_0 = torch.hann_window(self.win_length_new)
        self.register_buffer("hann_window_0", hann_window_0, persistent=False)

    def forward(self, audio: torch.Tensor, center=True):
        fft = torch.stft(
            audio,
            n_fft=self.n_fft_new,
            hop_length=self.hop_length_new,
            win_length=self.win_length_new,
            window=self.hann_window_0,
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec
