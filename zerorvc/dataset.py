import os
import numpy as np
import torch
import librosa
import logging
import shutil
from pkg_resources import resource_filename
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, Audio
from .preprocess import Preprocessor, crop_feats_length
from .hubert import HubertFeatureExtractor, HubertModel, load_hubert
from .f0 import F0Extractor, RMVPE, load_rmvpe
from .constants import *


logger = logging.getLogger(__name__)


def extract_hubert_features(
    rows,
    hfe: HubertFeatureExtractor,
    hubert: str | HubertModel | None,
    device: torch.device,
):
    if not hfe.is_loaded():
        model = load_hubert(hubert, device)
        hfe.load(model)
    feats = []
    for row in rows["wav_16k"]:
        feat = hfe.extract_feature_from(row["array"].astype("float32"))
        feats.append(feat)
    return {"hubert_feats": feats}


def extract_f0_features(
    rows, f0e: F0Extractor, rmvpe: str | RMVPE | None, device: torch.device
):
    if not f0e.is_loaded():
        model = load_rmvpe(rmvpe, device)
        f0e.load(model)
    f0s = []
    f0nsfs = []
    for row in rows["wav_16k"]:
        f0nsf, f0 = f0e.extract_f0_from(row["array"].astype("float32"))
        f0s.append(f0)
        f0nsfs.append(f0nsf)
    return {"f0": f0s, "f0nsf": f0nsfs}


def feature_postprocess(rows):
    phones = rows["hubert_feats"]
    for i, phone in enumerate(phones):
        phone = np.repeat(phone, 2, axis=0)
        n_num = min(phone.shape[0], 900)
        phone = phone[:n_num, :]
        phones[i] = phone

        if "f0" in rows:
            pitch = rows["f0"][i]
            pitch = pitch[:n_num]
            pitch = np.array(pitch, dtype=np.float32)
            rows["f0"][i] = pitch
        if "f0nsf" in rows:
            pitchf = rows["f0nsf"][i]
            pitchf = pitchf[:n_num]
            rows["f0nsf"][i] = pitchf
    return rows


def calculate_spectrogram(
    rows, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
):
    specs = []
    hann_window = np.hanning(win_length)
    pad_amount = int((win_length - hop_length) / 2)
    for row in rows["wav_gt"]:
        stft = librosa.stft(
            np.pad(row["array"], (pad_amount, pad_amount), mode="reflect"),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            center=False,
        )
        specs.append(np.abs(stft) + 1e-6)

    return {"spec": specs}


def fix_length(rows, hop_length=HOP_LENGTH):
    for i, row in enumerate(rows["spec"]):
        spec = np.array(row)
        phone = np.array(rows["hubert_feats"][i])
        pitch = np.array(rows["f0"][i])
        pitchf = np.array(rows["f0nsf"][i])
        wav_gt = np.array(rows["wav_gt"][i]["array"])

        spec, phone, pitch, pitchf = crop_feats_length(spec, phone, pitch, pitchf)

        phone_len = phone.shape[0]
        wav_gt = wav_gt[: phone_len * hop_length]

        rows["hubert_feats"][i] = phone
        rows["f0"][i] = pitch
        rows["f0nsf"][i] = pitchf
        rows["spec"][i] = spec
        rows["wav_gt"][i]["array"] = wav_gt
    return rows


def prepare(
    dir: str | DatasetDict,
    sr=SR_48K,
    hubert: str | HubertModel | None = None,
    rmvpe: str | RMVPE | None = None,
    batch_size=1,
    max_slice_length: float | None = 3.0,
    accelerator: Accelerator = None,
    include_mute=True,
    stage=3,
):
    """
    Prepare the dataset for training or evaluation.

    Args:
        dir (str | DatasetDict): The directory path or DatasetDict object containing the dataset.
        sr (int, optional): The target sampling rate. Defaults to SR_48K.
        hubert (str | HubertModel | None, optional): The Hubert model or its name to use for feature extraction. Defaults to None.
        rmvpe (str | RMVPE | None, optional): The RMVPE model or its name to use for feature extraction. Defaults to None.
        batch_size (int, optional): The batch size for processing the dataset. Defaults to 1.
        accelerator (Accelerator, optional): The accelerator object for distributed training. Defaults to None.
        include_mute (bool, optional): Whether to include a mute audio file in the directory dataset. Defaults to True.
        stage (int, optional): The dataset preparation level to perform. Defaults to 3. (Stage 1 and 3 are CPU intensive, Stage 2 is GPU intensive.)

    Returns:
        DatasetDict: The prepared dataset.
    """
    if accelerator is None:
        accelerator = Accelerator()

    if isinstance(dir, DatasetDict):
        ds = dir
    else:
        mute_source = resource_filename("zerorvc", "assets/mute/mute48k.wav")
        mute_dest = os.path.join(dir, "mute.wav")
        if include_mute and not os.path.exists(mute_dest):
            logger.info(f"Copying {mute_source} to {mute_dest}")
            shutil.copy(mute_source, mute_dest)

        ds: DatasetDict = load_dataset("audiofolder", data_dir=dir)

    for key in ds:
        ds[key] = ds[key].remove_columns(
            [col for col in ds[key].column_names if col != "audio"]
        )
    ds = ds.cast_column("audio", Audio(sampling_rate=sr))

    if stage <= 0:
        return ds

    # Stage 1, CPU intensive

    pp = Preprocessor(sr, max_slice_length) if max_slice_length is not None else None

    def preprocess(rows):
        wav_gt = []
        wav_16k = []
        for row in rows["audio"]:
            if pp is not None:
                slices = pp.preprocess_audio(row["array"])
                for slice in slices:
                    wav_gt.append({"path": "", "array": slice, "sampling_rate": sr})
                    slice16k = librosa.resample(slice, orig_sr=sr, target_sr=SR_16K)
                    wav_16k.append(
                        {"path": "", "array": slice16k, "sampling_rate": SR_16K}
                    )
            else:
                slice = row["array"]
                wav_gt.append({"path": "", "array": slice, "sampling_rate": sr})
                slice16k = librosa.resample(slice, orig_sr=sr, target_sr=SR_16K)
                wav_16k.append({"path": "", "array": slice16k, "sampling_rate": SR_16K})
        return {"wav_gt": wav_gt, "wav_16k": wav_16k}

    ds = ds.map(
        preprocess, batched=True, batch_size=batch_size, remove_columns=["audio"]
    )
    ds = ds.cast_column("wav_gt", Audio(sampling_rate=sr))
    ds = ds.cast_column("wav_16k", Audio(sampling_rate=SR_16K))

    if stage <= 1:
        return ds

    # Stage 2, GPU intensive

    hfe = HubertFeatureExtractor()
    ds = ds.map(
        extract_hubert_features,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"hfe": hfe, "hubert": hubert, "device": accelerator.device},
    )

    f0e = F0Extractor()
    ds = ds.map(
        extract_f0_features,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"f0e": f0e, "rmvpe": rmvpe, "device": accelerator.device},
    )

    if stage <= 2:
        return ds

    # Stage 3, CPU intensive

    ds = ds.map(feature_postprocess, batched=True, batch_size=batch_size)
    ds = ds.map(calculate_spectrogram, batched=True, batch_size=batch_size)
    ds = ds.map(fix_length, batched=True, batch_size=batch_size)

    return ds
