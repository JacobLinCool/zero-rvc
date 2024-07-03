from typing import Tuple
import numpy as np


def crop_feats_length(
    spec: np.ndarray, phone: np.ndarray, pitch: np.ndarray, pitchf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phone_len = phone.shape[0]
    spec_len = spec.shape[1]
    if phone_len != spec_len:
        len_min = min(phone_len, spec_len)
        phone = phone[:len_min, :]
        pitch = pitch[:len_min]
        pitchf = pitchf[:len_min]
        spec = spec[:, :len_min]
    return spec, phone, pitch, pitchf
