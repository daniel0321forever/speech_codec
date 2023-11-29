import torch
from torch import nn, Tensor
import math

"""
pitch input shape:

(80, 1) -> (frames, pitch)

encode the pitch at each frame into positional encoded vector
"""

def positional_encoding(pitch_array: torch.Tensor):
    seq_len = 4
    n = 10000

    even = lambda k, i: math.sin(k / (n ** (2 * i / seq_len)))
    odd = lambda k, i: math.cos(k / (n ** (2 * i / seq_len)))

    pe = []
    for k in pitch_array:
        pe.append([])
        k = k.item()
        
        for idx in range(seq_len):
            i = idx //2

            p_ki = even(k, i) if idx % 2 == 0 else odd(k, i)
            pe[-1].append(p_ki)

    pe = torch.tensor(pe)

    return pe

tensor = torch.tensor(
    [[0], [1], [2], [3]]
)

print(positional_encoding(tensor))

