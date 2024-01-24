import logging, os, sys

import torch
from torch import nn
import math
import numpy as np
import librosa

sample_rate = 24000
fft_size = 2048
win_length = 1200
hop_length = 300
supp_fft_size = 1024
supp_hop_length = 300

num_mels = 80
fmin = 80
fmax = 7600

import h5py
def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 80):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def compute_mel(y):
    
    eps = 1e-10
    log_base = 10.0

    x_stft = librosa.stft(
        y=y,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis -> project the freq at each time frame onto the mel basis (not log yet)
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))


    x_stft_supp = librosa.stft(
        y=y,
        n_fft=supp_fft_size,
        hop_length=supp_hop_length,
        win_length=supp_fft_size,
        center=True,
        pad_mode="reflect",
    )
    # just spectrum, no mel cepstrum
    spc_supp = np.abs(x_stft_supp).T

    # Get log spec
    # spc = librosa.amplitude_to_db(spc)

    # print (mel.shape)
    return np.log10(mel), spc_supp

def compute_pitch(y):
    sr = 24000
    pitch, mag = librosa.piptrack(
        y=y, 
        sr=sr, 
        n_fft=fft_size, 
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        pad_mode="reflect",
    )

    pitch = pitch.T
    mag = mag.T

    return pitch, mag
