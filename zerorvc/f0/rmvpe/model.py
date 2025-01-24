import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huggingface_hub import PyTorchModelHubMixin
from .seq import BiGRU
from .deepunet import DeepUnet
from .mel import MelSpectrogram
from .constants import *

logger = logging.getLogger(__name__)


class RMVPE(nn.Module, PyTorchModelHubMixin):
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
        self.cents_mapping_torch = torch.from_numpy(self.cents_mapping).to(
            dtype=torch.float32
        )

    def to(self, device):
        self.cents_mapping_torch = self.cents_mapping_torch.to(device)
        return super().to(device)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

    def mel2hidden(self, mel: torch.Tensor):
        with torch.no_grad():
            n_frames = mel.shape[2]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            mel = F.pad(mel, (0, n_pad), mode="constant")
            hidden = self(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden: torch.Tensor, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer(self, audio: torch.Tensor, thred=0.03, return_tensor=False):
        mel = self.mel_extractor(audio.unsqueeze(0))
        hidden = self.mel2hidden(mel)
        hidden = hidden[0].float()
        f0 = self.decode(hidden, thred=thred)
        if return_tensor:
            return f0
        return f0.cpu().numpy()

    def infer_from_audio(self, audio: np.ndarray, thred=0.03):
        audio = torch.from_numpy(audio).to(next(self.parameters()).device)
        return self.infer(audio, thred=thred)

    def to_local_average_cents(
        self, salience: torch.Tensor, thred=0.05
    ) -> torch.Tensor:
        center = torch.argmax(salience, dim=1)
        salience = F.pad(salience, (4, 4))

        center += 4
        batch_indices = torch.arange(salience.shape[0], device=salience.device)

        # Create indices for the 9-point window around each center
        offsets = torch.arange(-4, 5, device=salience.device)
        indices = center.unsqueeze(1) + offsets.unsqueeze(0)

        # Extract values using advanced indexing
        todo_salience = salience[batch_indices.unsqueeze(1), indices]
        todo_cents_mapping = self.cents_mapping_torch[indices]

        product_sum = torch.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = torch.sum(todo_salience, 1)
        divided = product_sum / weight_sum

        maxx = torch.max(salience, 1).values
        divided[maxx <= thred] = 0

        return divided
