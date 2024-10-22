from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from accelerate import Accelerator
from datasets import Dataset

from .f0 import F0Extractor, RMVPE, load_rmvpe
from .hubert import HubertFeatureExtractor, HubertModel, load_hubert
from .synthesizer import SynthesizerTrnMs768NSFsid
from .constants import *

logger = getLogger(__name__)


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
        name: str | SynthesizerTrnMs768NSFsid,
        sr=SR_48K,
        segment_size=30.0,
        hubert: str | HubertModel | None = None,
        rmvpe: str | RMVPE | None = None,
        accelerator: Accelerator = Accelerator(),
        from_pretrained_kwargs={},
    ):
        """
        Initializes an instance of the RVC class.

        Args:
            name (str | SynthesizerTrnMs768NSFsid): The name of the pre-trained model or the model instance itself.
            sr (int, optional): The sample rate of the input audio. Defaults to SR_48K.
            hubert (str | HubertModel | None, optional): The name of the pre-trained Hubert model or the model instance itself. Defaults to None.
            rmvpe (str | RMVPE | None, optional): The name of the pre-trained RMVPE model or the model instance itself. Defaults to None.
            accelerator (Accelerator, optional): The accelerator device for model inference. Defaults to Accelerator().
            from_pretrained_kwargs (dict, optional): Additional keyword arguments for loading the pre-trained model. Defaults to {}.
        """
        self.model = (
            SynthesizerTrnMs768NSFsid.from_pretrained(name, **from_pretrained_kwargs)
            if isinstance(name, str)
            else name
        )
        self.model = self.model.to(accelerator.device)
        self.sr = sr
        self.segment_size = segment_size
        self.hubert = HubertFeatureExtractor(load_hubert(hubert, accelerator.device))
        self.rmvpe = F0Extractor(load_rmvpe(rmvpe, accelerator.device))
        self.accelerator = accelerator

    @staticmethod
    def from_pretrained(
        name: str,
        sr=SR_48K,
        segment_size=30.0,
        hubert: str | HubertModel | None = None,
        rmvpe: str | RMVPE | None = None,
        accelerator: Accelerator = Accelerator(),
        **from_pretrained_kwargs,
    ):
        """
        Creates an instance of RVC using the from_pretrained method.

        Args:
            name (str): The name of the pre-trained model.
            sr (int, optional): The sample rate of the input audio. Defaults to SR_48K.
            segment_size (float, optional): The segment size for splitting the input audio. Defaults to 30.0 seconds.
            hubert (str | HubertModel | None, optional): The name of the pre-trained Hubert model or the model instance itself. Defaults to None.
            rmvpe (str | RMVPE | None, optional): The name of the pre-trained RMVPE model or the model instance itself. Defaults to None.
            accelerator (Accelerator, optional): The accelerator device for model inference. Defaults to Accelerator().
            from_pretrained_kwargs (dict): Additional keyword arguments for loading the pre-trained model.

        Returns:
            RVC: An instance of the RVC class.
        """
        return RVC(
            name, sr, segment_size, hubert, rmvpe, accelerator, from_pretrained_kwargs
        )

    def convert(
        self, audio: str | Dataset | np.ndarray, protect=0.33, pitch_modification=0.0
    ):
        """
        Converts the input audio to the target voice using the pre-trained model.

        Args:
            audio (str | Dataset | np.ndarray): The input audio to be converted. It can be a file path, a dataset of audio samples, or a numpy array.
            protect (float, optional): The protection factor for preserving the original voice. Defaults to 0.33.
            pitch_modification (float, optional): The pitch modification factor. Defaults to 0.0.

        Returns:
            np.ndarray: The converted audio in the target voice.
            If the input is a dataset, it yields the converted audio samples one by one.
        """
        logger.info(
            f"audio: {audio}, protect: {protect}, pitch_modification: {pitch_modification}"
        )
        if isinstance(audio, str):
            return self.convert_file(audio, protect, pitch_modification)
        if isinstance(audio, Dataset):
            return self.convert_dataset(audio, protect, pitch_modification)
        return self.convert_from_wav16k(audio, protect, pitch_modification)

    def convert_dataset(self, dataset: Dataset, protect=0.33, pitch_modification=0.0):
        """
        Converts a dataset of audio samples to the target voice using the pre-trained model.

        Args:
            dataset (Dataset): The dataset of audio samples to be converted.
            protect (float, optional): The protection factor for preserving the original voice. Defaults to 0.33.
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
                phone, pitchf, pitch, protect, pitch_modification
            )

    def convert_file(
        self, audio: str, protect=0.33, pitch_modification=0.0
    ) -> np.ndarray:
        """
        Converts a single audio file to the target voice using the pre-trained model.

        Args:
            audio (str): The path to the audio file to be converted.
            protect (float, optional): The protection factor for preserving the original voice. Defaults to 0.33.
            pitch_modification (float, optional): The pitch modification factor. Defaults to 0.0.

        Returns:
            np.ndarray: The converted audio in the target voice.
        """
        wav16k, _ = librosa.load(audio, sr=SR_16K)
        logger.info(f"Loaded {audio} with shape {wav16k.shape}")
        return self.convert_from_wav16k(wav16k, protect, pitch_modification)

    def convert_from_wav16k(
        self, wav16k: np.ndarray, protect=0.33, pitch_modification=0.0
    ) -> np.ndarray:
        """
        Converts a 16kHz waveform to the target voice using the pre-trained model.

        Args:
            wav16k (np.ndarray): The 16kHz waveform to be converted.
            protect (float, optional): The protection factor for preserving the original voice. Defaults to 0.33.
            pitch_modification (float, optional): The pitch modification factor. Defaults to 0.0.

        Returns:
            np.ndarray: The converted audio in the target voice.
        """

        ret = []
        segment_size = int(self.segment_size * SR_16K)
        for i in range(0, len(wav16k), segment_size):
            segment = wav16k[i : i + segment_size]
            segment = np.pad(segment, (SR_16K, SR_16K), mode="reflect")
            logger.info(f"Padded audio with shape {segment.shape}")

            pitchf, pitch = self.rmvpe.extract_f0_from(segment)
            phone = self.hubert.extract_feature_from(segment)

            ret.append(
                self.convert_from_features(
                    phone, pitchf, pitch, protect, pitch_modification
                )[self.sr : -self.sr]
            )

        return np.concatenate(ret)

    def convert_from_features(
        self,
        phone: np.ndarray,
        pitchf: np.ndarray,
        pitch: np.ndarray,
        protect=0.33,
        pitch_modification=0.0,
    ) -> np.ndarray:
        """
        Converts audio features (phone, pitchf, pitch) to the target voice using the pre-trained model.

        Args:
            phone (np.ndarray): The phone features of the audio.
            pitchf (np.ndarray): The pitch features of the audio.
            pitch (np.ndarray): The pitch values of the audio.
            protect (float, optional): The protection factor for preserving the original voice. Defaults to 0.33.
            pitch_modification (float, optional): The pitch modification factor. Defaults to 0.0.

        Returns:
            np.ndarray: The converted audio in the target voice.
        """
        use_protect = protect < 0.5

        if not np.isclose(pitch_modification, 0.0):
            pitchf *= pow(2, pitch_modification / 12)
            pitch = self.rmvpe.calculate_f0_from_f0nsf(pitchf)

        pitchf = np.expand_dims(pitchf, axis=0)
        pitch = np.expand_dims(pitch, axis=0)
        phone = np.expand_dims(phone, axis=0)

        self.model.eval()
        with torch.no_grad(), self.accelerator.device:
            pitchf = torch.from_numpy(pitchf).to(
                dtype=torch.float32, device=self.accelerator.device
            )
            pitch = torch.from_numpy(pitch).to(
                dtype=torch.long, device=self.accelerator.device
            )
            phone = torch.from_numpy(phone).to(
                dtype=torch.float32, device=self.accelerator.device
            )

            if use_protect:
                feats0 = phone.clone()

            feats: torch.Tensor = F.interpolate(
                phone.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)
            if use_protect:
                feats0: torch.Tensor = F.interpolate(
                    feats0.permute(0, 2, 1), scale_factor=2
                ).permute(0, 2, 1)

            # It's originally like this, but I think it's ok to assume that feats.shape[1] <= phone_len
            # maybe we should use the same crop function from preprocessor
            # phone_len = wav16k.shape[0] // 160
            # if feats.shape[1] < phone_len:
            # ...
            phone_len = feats.shape[1]
            pitch = pitch[:, :phone_len]
            pitchf = pitchf[:, :phone_len]

            if use_protect:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)
                feats = feats * pitchff + feats0 * (1 - pitchff)
                feats = feats.to(feats0.dtype)

            phone_len = torch.tensor([phone_len], dtype=torch.long)
            sid = torch.tensor([0], dtype=torch.long)

            logger.info(f"Feats shape: {feats.shape}")
            logger.info(f"Phone len: {phone_len}")
            logger.info(f"Pitch shape: {pitch.shape}")
            logger.info(f"Pitchf shape: {pitchf.shape}")
            logger.info(f"SID shape: {sid}")
            audio_segment = (
                self.model.infer(feats, phone_len, pitch, pitchf, sid)[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
            logger.info(
                f"Generated audio shape: {audio_segment.shape} {audio_segment.dtype}"
            )
            return audio_segment
