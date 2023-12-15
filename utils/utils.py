import torch
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


def positional_encoding(pitch_array: torch.Tensor, dim=64, is_batch = True):
    """
    returns
        (batch_size, frames, seq_len)
    
    params
        - pitch_array: an shape (batch_size, frames, 1) array that include the pitch number in each frame of a batch
    """

    seq_len = dim
    n = 10000

    even = lambda k, i: math.sin(k / (n ** (2 * i / seq_len)))
    odd = lambda k, i: math.cos(k / (n ** (2 * i / seq_len)))

    pe = []

    if is_batch == False:
        for k in pitch_array:
            pe.append([])
            k = k.item()

            for idx in range(seq_len):
                i = idx // 2

                p_ki = even(k, i) if idx % 2 == 0 else odd(k, i)
                pe[-1].append(p_ki)
        
        pe = torch.tensor(pe)

        return pe

    for batch in pitch_array:
        pe.append([])
        for k in batch:
            pe[-1].append([])
            k = k.item()
            
            for idx in range(seq_len):
                i = idx //2

                p_ki = even(k, i) if idx % 2 == 0 else odd(k, i)
                pe[-1][-1].append(p_ki)

    pe = torch.tensor(pe)

    return pe

def positional_encoding_by_frame(pitch_array: torch.Tensor):
    """
    returns
        (batch_size, frames, seq_len + 1)
    
    params
        - pitch_array: an shape (batch_size, frames, 1) array that include the pitch number in each frame of a batch
    """

    seq_len = 64
    n = 10000

    even = lambda k, i: math.sin(k / (n ** (2 * i / seq_len)))
    odd = lambda k, i: math.cos(k / (n ** (2 * i / seq_len)))

    encoded_array = []
    for batch in pitch_array:
        encoded_array.append([])
        for k in range(len(batch)):
            encoded_array[-1].append([])
            
            pitch = batch[k].item()
            encoded_array[-1][-1].append(pitch)

            for idx in range(seq_len):
                i = idx //2
                p_ki = even(k, i) if idx % 2 == 0 else odd(k, i)
                encoded_array[-1][-1].append(p_ki)

    encoded_array = torch.tensor(encoded_array)

    return encoded_array

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
