import torch
import math

def positional_encoding(pitch_array: torch.Tensor):
    """
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