import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from librosa.util import pad_center
from scipy.signal import get_window


class STFT(nn.Module):
    def __init__(
        self, filter_length=1024, hop_length=512, win_length=None, window="hann"
    ):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        assert filter_length >= self.win_length
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T

        self.register_buffer("forward_basis", forward_basis.float(), persistent=False)
        self.register_buffer("inverse_basis", inverse_basis.float(), persistent=False)
        self.register_buffer("fft_window", fft_window.float(), persistent=False)

    def transform(self, input_data, return_phase=False):
        """Take input data (audio) to STFT domain using convolution."""
        input_data = F.pad(
            input_data,
            (self.pad_amount, self.pad_amount),
            mode="reflect",
        )

        # Reshape input for convolution
        input_data = input_data.unsqueeze(1)

        # Create windowed basis as convolution weights
        forward_transform = F.conv1d(
            input_data,
            self.forward_basis.unsqueeze(1),
            stride=self.hop_length,
            groups=1,
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)

        if return_phase:
            phase = torch.atan2(imag_part.data, real_part.data)
            return magnitude, phase
        else:
            return magnitude

    def inverse(self, magnitude, phase):
        """Inverse STFT using transposed convolution."""
        recombined = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        # Apply inverse basis using transposed convolution
        inverse_transform = F.conv_transpose1d(
            recombined,
            self.inverse_basis.unsqueeze(1),
            stride=self.hop_length,
            groups=1,
        )

        # Remove padding
        inverse_transform = inverse_transform[:, 0, self.pad_amount : -self.pad_amount]

        # Window normalization
        n_frames = magnitude.size(-1)
        window_sum = F.conv_transpose1d(
            torch.ones_like(magnitude),
            (self.fft_window * self.fft_window).unsqueeze(0).unsqueeze(0),
            stride=self.hop_length,
        )
        window_sum = window_sum[:, 0, self.pad_amount : -self.pad_amount]
        inverse_transform = inverse_transform / (window_sum + 1e-6)

        return inverse_transform

    def forward(self, input_data):
        """Take input data (audio) to STFT domain and then back to audio.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        self.magnitude, self.phase = self.transform(input_data, return_phase=True)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
