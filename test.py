import torch
import math
from torch import nn, tensor, Tensor

bos = torch.zeros(16, 1, 256)
x  = torch.ones(16, 80, 256)

concat = torch.concat([bos, x[:, :-1]], dim=1)
print(concat.shape)