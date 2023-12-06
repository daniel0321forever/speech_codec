import torch
import math

def positional_encoding(pitch_array: torch.Tensor):
    """
    returns
        (batch_size, frames, seq_len)
    
    params
        - pitch_array: an shape (batch_size, frames, 1) array that include the pitch number in each frame of a batch
    """
    seq_len = 64
    n = 10000

    even = lambda k, i: math.sin(k / (n ** (2 * i / seq_len)))
    odd = lambda k, i: math.cos(k / (n ** (2 * i / seq_len)))

    pe = []
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
