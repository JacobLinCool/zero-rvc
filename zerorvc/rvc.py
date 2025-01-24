from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from accelerate import Accelerator
from datasets import Dataset

from .f0 import F0Extractor, RMVPE, load_rmvpe
from .hubert import HubertFeatureExtractor, HubertModel, load_hubert
from .synthesizer import SynthesizerTrnMs768NSFsid
from .constants import *

logger = getLogger(__name__)


class Synthesizer(SynthesizerTrnMs768NSFsid):
    def forward(self, phone, pitch, pitchf, sid):
        if type(phone.shape[1]) == int:
            phone_lengths = torch.tensor(
                [phone.shape[1]], device=phone.device, dtype=torch.int32
            )
        else:
            phone_lengths = phone.shape[1]
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, pitchf, g=g, n_res=None)
        return o


class FeatureExtractor(nn.Module):
    def __init__(self, hubert: HubertModel, rmvpe: RMVPE):
        super().__init__()
        self.hubert = hubert
        self.rmvpe = rmvpe

    def to(self, device):
        self.hubert = self.hubert.to(device)
        self.rmvpe = self.rmvpe.to(device)
        return super().to(device)

    def forward(self, audio16k, pitch_modification):
        phone = self.hubert(audio16k, output_hidden_states=True)["hidden_states"][12]
        phone = phone.squeeze(0).float()
        phone_lengths = phone.shape[0]
        if type(phone_lengths) == int:
            phone_lengths = torch.tensor(
                [phone_lengths], device=phone.device, dtype=torch.int32
            )

        pitchf = self.rmvpe.infer(audio16k.squeeze(0), thred=0.03, return_tensor=True)
        pitchf *= torch.pow(
            2,
            torch.tensor(
                pitch_modification / 12.0, dtype=torch.float32, device=pitchf.device
            ),
        )
        pitch = self.calculate_f0_from_f0nsf_torch(pitchf)

        pitch = pitch.unsqueeze(0)
        pitchf = pitchf.unsqueeze(0)
        phone = phone.unsqueeze(0)
        logger.info(
            f"{phone.shape=}, {phone_lengths=}, {pitch.shape=}, {pitchf.shape=}"
        )

        feats0 = phone.clone()
        feats: torch.Tensor = F.interpolate(
            phone.permute(0, 2, 1), scale_factor=2
        ).permute(0, 2, 1)
        feats0: torch.Tensor = F.interpolate(
            feats0.permute(0, 2, 1), scale_factor=2
        ).permute(0, 2, 1)

        phone_len = feats.shape[1]
        pitch = pitch[:, :phone_len]
        pitchf = pitchf[:, :phone_len]

        pitchff = pitchf.clone()
        pitchff[pitchf > 0] = 1
        pitchff[pitchf < 1] = 0.33
        pitchff = pitchff.unsqueeze(-1)
        feats = feats * pitchff + feats0 * (1 - pitchff)
        feats = feats.to(feats0.dtype)

        if type(phone_len) == int:
            phone_len = torch.tensor(
                [phone_len], device=feats.device, dtype=torch.int32
            )
        else:
            phone_len = phone_len.unsqueeze(0)

        logger.info(f"{feats.shape=}, {pitch.shape=}, {pitchf.shape=}, {phone_len=}")
        return feats, phone_len, pitch, pitchf

    def calculate_f0_from_f0nsf_torch(self, f0nsf: torch.Tensor):
        f0_mel = 1127 * torch.log(1 + f0nsf / 700)
        f0_max = torch.tensor(1100.0)
        f0_min = torch.tensor(50.0)
        f0_bin = torch.tensor(256)
        f0_mel_max = 1127 * torch.log(1 + f0_max / 700)
        f0_mel_min = 1127 * torch.log(1 + f0_min / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
            f0_mel_max - f0_mel_min
        ) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
        f0 = torch.round(f0_mel).long()
        f0 = torch.clamp(f0, 1, 255)

        return f0


class RVC:
    """
    RVC (Retrieval-based Voice Conversion) class for converting speech using a pre-trained model.

    Args:
        name (str | SynthesizerTrnMs768NSFsid): The name of the pre-trained model or the model instance itself.
        sr (int, optional): The sample rate of the input audio. Defaults to SR_48K.
        segment_size (float, optional): The segment size for splitting the input audio. Defaults to 30.0 seconds.
        hubert (str | HubertModel | None, optional): The name of the pre-trained Hubert model or the model instance itself. Defaults to None.
        rmvpe (str | RMVPE | None, optional): The name of the pre-trained RMVPE model or the model instance itself. Defaults to None.
        accelerator (Accelerator, optional): The accelerator device for model inference. Defaults to Accelerator().
        from_pretrained_kwargs (dict, optional): Additional keyword arguments for loading the pre-trained model. Defaults to {}.

    Methods:
        from_pretrained(name, sr=SR_48K, hubert=None, rmvpe=None, accelerator=Accelerator(), **from_pretrained_kwargs):
            Creates an instance of RVC using the from_pretrained method.

        convert(audio, protect=0.33):
            Converts the input audio to the target voice using the pre-trained model.

        convert_dataset(dataset, protect=0.33):
            Converts a dataset of audio samples to the target voice using the pre-trained model.

        convert_file(audio, protect=0.33):
            Converts a single audio file to the target voice using the pre-trained model.

        convert_from_wav16k(wav16k, protect=0.33):
            Converts a 16kHz waveform to the target voice using the pre-trained model.

        convert_from_features(phone, pitchf, pitch, protect=0.33):
            Converts audio features (phone, pitchf, pitch) to the target voice using the pre-trained model.
    """

    def __init__(
        self,
        synthesizer: str | Synthesizer,
        hubert: HubertModel | None = None,
        rmvpe: RMVPE | None = None,
        sr=SR_48K,
        segment_size=30.0,
        accelerator: Accelerator | None = None,
        from_pretrained_kwargs={},
    ):
        """
        Initializes an instance of the RVC class.

        Args:
            synthesizer (str | Synthesizer): The name of the pre-trained model or the model instance itself.
            hubert (str | HubertModel | None, optional): The name of the pre-trained Hubert model or the model instance itself. Defaults to None.
            rmvpe (str | RMVPE | None, optional): The name of the pre-trained RMVPE model or the model instance itself. Defaults to None.
            sr (int, optional): The sample rate of the input audio. Defaults to SR_48K.
            segment_size (float, optional): The segment size for splitting the input audio. Defaults to 30.0 seconds.
            accelerator (Accelerator, optional): The accelerator device for model inference. Defaults to Accelerator().
            from_pretrained_kwargs (dict, optional): Additional keyword arguments for loading the pre-trained model. Defaults to {}.
        """
        accelerator = accelerator or Accelerator()
        self.accelerator = accelerator

        self.synthesizer = (
            Synthesizer.from_pretrained(synthesizer, **from_pretrained_kwargs)
            if isinstance(synthesizer, str)
            else synthesizer
        )
        self.synthesizer = self.synthesizer.to(accelerator.device)

        hubert = hubert or load_hubert()
        rmvpe = rmvpe or load_rmvpe()
        self.feature_extractor = FeatureExtractor(hubert, rmvpe)
        self.feature_extractor = self.feature_extractor.to(accelerator.device)

        self.sr = sr
        self.segment_size = segment_size

    @staticmethod
    def from_pretrained(
        name: str,
        hubert: HubertModel | None = None,
        rmvpe: RMVPE | None = None,
        sr=SR_48K,
        segment_size=30.0,
        accelerator: Accelerator | None = None,
        **from_pretrained_kwargs,
    ):
        """
        Creates an instance of RVC using the from_pretrained method.

        Args:
            name (str): The name of the pre-trained model.
            hubert (HubertModel | None, optional): The name of the pre-trained Hubert model or the model instance itself. Defaults to None.
            rmvpe (RMVPE | None, optional): The name of the pre-trained RMVPE model or the model instance itself. Defaults to None.
            sr (int, optional): The sample rate of the input audio. Defaults to SR_48K.
            segment_size (float, optional): The segment size for splitting the input audio. Defaults to 30.0 seconds.
            accelerator (Accelerator, optional): The accelerator device for model inference. Defaults to Accelerator().
            from_pretrained_kwargs (dict): Additional keyword arguments for loading the pre-trained model.

        Returns:
            RVC: An instance of the RVC class.
        """
        return RVC(
            name,
            hubert=hubert,
            rmvpe=rmvpe,
            sr=sr,
            segment_size=segment_size,
            accelerator=accelerator,
            from_pretrained_kwargs=from_pretrained_kwargs,
        )

    def convert(self, audio: str | Dataset | np.ndarray, pitch_modification=0.0):
        """
        Converts the input audio to the target voice using the pre-trained model.

        Args:
            audio (str | Dataset | np.ndarray): The input audio to be converted. It can be a file path, a dataset of audio samples, or a numpy array.
            pitch_modification (float, optional): The pitch modification factor. Defaults to 0.0.

        Returns:
            np.ndarray: The converted audio in the target voice.
            If the input is a dataset, it yields the converted audio samples one by one.
        """
        logger.info(f"audio: {audio}, pitch_modification: {pitch_modification}")
        if isinstance(audio, str):
            return self.convert_file(audio, pitch_modification=pitch_modification)
        if isinstance(audio, Dataset):
            return self.convert_dataset(audio, pitch_modification=pitch_modification)
        return self.convert_from_wav16k(audio, pitch_modification=pitch_modification)

    def convert_dataset(self, dataset: Dataset, pitch_modification=0.0):
        """
        Converts a dataset of audio samples to the target voice using the pre-trained model.

        Args:
            dataset (Dataset): The dataset of audio samples to be converted.
            pitch_modification (float, optional): The pitch modification factor. Defaults to 0.0.

        Yields:
            np.ndarray: The converted audio samples in the target voice.
        """
        for i, data in enumerate(dataset):
            logger.info(f"Converting data {i}")
            phone = data["hubert_feats"]
            pitchf = data["f0nsf"]
            pitch = data["f0"]
            yield self.convert_from_features(
                phone=phone,
                pitchf=pitchf,
                pitch=pitch,
                pitch_modification=pitch_modification,
            )

    def convert_file(self, audio: str, pitch_modification=0.0) -> np.ndarray:
        """
        Converts a single audio file to the target voice using the pre-trained model.

        Args:
            audio (str): The path to the audio file to be converted.
            pitch_modification (float, optional): The pitch modification factor. Defaults to 0.0.

        Returns:
            np.ndarray: The converted audio in the target voice.
        """
        wav16k, _ = librosa.load(audio, sr=SR_16K)
        logger.info(f"Loaded {audio} with shape {wav16k.shape}")
        return self.convert_from_wav16k(wav16k, pitch_modification=pitch_modification)

    @torch.no_grad()
    def convert_from_wav16k(
        self, wav16k: np.ndarray, pitch_modification=0.0
    ) -> np.ndarray:
        """
        Converts a 16kHz waveform to the target voice using the pre-trained model.

        Args:
            wav16k (np.ndarray): The 16kHz waveform to be converted.
            pitch_modification (float, optional): The pitch modification factor. Defaults to 0.0.

        Returns:
            np.ndarray: The converted audio in the target voice.
        """
        self.feature_extractor.eval()
        feature_extractor_device = next(self.feature_extractor.parameters()).device

        ret = []
        segment_size = int(self.segment_size * SR_16K)
        for i in range(0, len(wav16k), segment_size):
            segment = wav16k[i : i + segment_size]
            segment = np.pad(segment, (SR_16K, SR_16K), mode="reflect")
            logger.info(f"Padded audio with shape {segment.shape}")

            phone, phone_lengths, pitch, pitchf = self.feature_extractor(
                torch.from_numpy(segment)
                .unsqueeze(0)
                .to(device=feature_extractor_device),
                pitch_modification,
            )
            print(f"{phone.shape=}, {phone_lengths=}, {pitch.shape=}, {pitchf.shape=}")

            ret.append(
                self.convert_from_features(phone, pitchf, pitch)[self.sr : -self.sr]
            )

        return np.concatenate(ret)

    @torch.no_grad()
    def convert_from_features(
        self,
        phone: np.ndarray | torch.Tensor,
        pitchf: np.ndarray | torch.Tensor,
        pitch: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Converts audio features (phone, pitchf, pitch) to the target voice using the pre-trained model.

        Args:
            phone (np.ndarray): The phone features of the audio.
            pitchf (np.ndarray): The pitch features of the audio.
            pitch (np.ndarray): The pitch values of the audio.

        Returns:
            np.ndarray: The converted audio in the target voice.
        """
        self.synthesizer.eval()
        synthesizer_device = next(self.synthesizer.parameters()).device

        if isinstance(phone, np.ndarray):
            phone = torch.from_numpy(phone).to(device=synthesizer_device)
        if isinstance(pitchf, np.ndarray):
            pitchf = torch.from_numpy(pitchf).to(device=synthesizer_device)
        if isinstance(pitch, np.ndarray):
            pitch = torch.from_numpy(pitch).to(device=synthesizer_device)

        if phone.dim() == 2:
            phone = phone.unsqueeze(0)
        if pitchf.dim() == 1:
            pitchf = pitchf.unsqueeze(0)
        if pitch.dim() == 1:
            pitch = pitch.unsqueeze(0)

        sid = torch.tensor([0], device=synthesizer_device, dtype=torch.int32)

        audio_segment = (
            self.synthesizer(phone, pitch, pitchf, sid).squeeze().cpu().float().numpy()
        )
        logger.info(
            f"Generated audio shape: {audio_segment.shape} {audio_segment.dtype}"
        )
        return audio_segment
