import torch
from torch import nn, Tensor
import math
from utils import positional_encoding

"""
pitch input shape:

(80, 1) -> (frames, pitch)

encode the pitch at each frame into positional encoded vector
"""


tensor = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).view([2, 4, 1])
print(positional_encoding(tensor))

