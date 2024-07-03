import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .seq import BiGRU
from .deepunet import DeepUnet
from .mel import MelSpectrogram
from .constants import *

logger = logging.getLogger(__name__)


class RMVPE(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: int,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.device = torch.device("cpu")
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, None, MEL_FMIN, MEL_FMAX
        )
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

        cents_mapping = 20 * np.arange(360) + MAGIC_CONST
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

    def to(self, device):
        self.device = device
        return super().to(device)

    def mel2hidden(self, mel: torch.Tensor):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            # mel = mel.half() if self.is_half else mel.float()
            hidden = self(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden: np.ndarray, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        return f0

    def infer(self, audio: torch.Tensor, thred=0.03):
        mel = self.mel_extractor(audio.unsqueeze(0), center=True)
        hidden = self.mel2hidden(mel)
        hidden = hidden[0]
        f0 = self.decode(hidden.float().cpu(), thred=thred)
        return f0

    def infer_from_audio(self, audio: np.ndarray, thred=0.03):
        audio = torch.from_numpy(audio).to(self.device)
        return self.infer(audio, thred=thred)

    def to_local_average_cents(self, salience: np.ndarray, thred=0.05) -> np.ndarray:
        center = np.argmax(salience, axis=1)  # 帧长#index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # 帧长,368

        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])

        todo_salience = np.array(todo_salience)  # 帧长，9
        todo_cents_mapping = np.array(todo_cents_mapping)  # 帧长，9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # 帧长
        devided = product_sum / weight_sum  # 帧长

        maxx = np.max(salience, axis=1)  # 帧长
        devided[maxx <= thred] = 0
        return devided
