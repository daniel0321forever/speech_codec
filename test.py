import torch
from torch import nn, tensor

a = tensor([[0.1, 0.4, 0.5], [0.3, 0.6, 0.8], [0.1, 0.2, 0.1]])
print(a.max(1)) # values, indices