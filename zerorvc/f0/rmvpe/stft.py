import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from librosa.util import pad_center
from scipy.signal import get_window


class TorchSTFT(nn.Module):
    def __init__(
        self, filter_length=1024, hop_length=512, win_length=None, window="hann"
    ):
        """
        This module implements an STFT using PyTorch's stft function.

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
                (default: {'hann'})
        """
        super(TorchSTFT, self).__init__()
        self.n_fft_new = filter_length
        self.hop_length_new = hop_length
        self.win_length_new = win_length if win_length else filter_length
        self.center = True
        hann_window_0 = torch.hann_window(self.win_length_new)
        self.register_buffer("hann_window_0", hann_window_0, persistent=False)

    def forward(self, input_data):
        fft = torch.stft(
            input_data,
            n_fft=self.n_fft_new,
            hop_length=self.hop_length_new,
            win_length=self.win_length_new,
            window=self.hann_window_0,
            center=self.center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        return magnitude


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

    def forward(self, input_data):
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

        return magnitude
