import logging

import torch
import torch.utils.data

logger = logging.getLogger(__name__)


class TextAudioCollateMultiNSFsid:
    """Zero-pads model inputs and targets"""

    def __init__(self):
        pass

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        device = batch[0]["spec"].device

        with device:
            # Right zero-pad all one-hot text sequences to max input length
            _, ids_sorted_decreasing = torch.sort(
                torch.tensor([x["spec"].size(1) for x in batch], dtype=torch.long),
                dim=0,
                descending=True,
            )

            max_spec_len = max([x["spec"].size(1) for x in batch])
            max_wave_len = max([x["wav_gt"]["array"].size(0) for x in batch])
            spec_lengths = torch.zeros(len(batch), dtype=torch.long)
            wave_lengths = torch.zeros(len(batch), dtype=torch.long)
            spec_padded = torch.zeros(
                len(batch), batch[0]["spec"].size(0), max_spec_len, dtype=torch.float32
            )
            wave_padded = torch.zeros(len(batch), 1, max_wave_len, dtype=torch.float32)

            max_phone_len = max([x["hubert_feats"].size(0) for x in batch])
            phone_lengths = torch.zeros(len(batch), dtype=torch.long)
            phone_padded = torch.zeros(
                len(batch),
                max_phone_len,
                batch[0]["hubert_feats"].shape[1],
                dtype=torch.float32,
            )  # (spec, wav, phone, pitch)
            pitch_padded = torch.zeros(len(batch), max_phone_len, dtype=torch.long)
            pitchf_padded = torch.zeros(len(batch), max_phone_len, dtype=torch.float32)
            # dv = torch.FloatTensor(len(batch), 256)#gin=256
            sid = torch.zeros(len(batch), dtype=torch.long)

            for i in range(len(ids_sorted_decreasing)):
                row = batch[ids_sorted_decreasing[i]]

                spec = row["spec"]
                spec_padded[i, :, : spec.size(1)] = spec
                spec_lengths[i] = spec.size(1)

                wave = row["wav_gt"]["array"]
                wave_padded[i, :, : wave.size(0)] = wave
                wave_lengths[i] = wave.size(0)

                phone = row["hubert_feats"]
                phone_padded[i, : phone.size(0), :] = phone
                phone_lengths[i] = phone.size(0)

                pitch = row["f0"]
                pitch_padded[i, : pitch.size(0)] = pitch
                pitchf = row["f0nsf"]
                pitchf_padded[i, : pitchf.size(0)] = pitchf

                sid[i] = torch.tensor([0], dtype=torch.long)

            return (
                phone_padded,
                phone_lengths,
                pitch_padded,
                pitchf_padded,
                spec_padded,
                spec_lengths,
                wave_padded,
                wave_lengths,
                sid,
            )
