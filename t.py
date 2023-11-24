import torch
from torch import nn, Tensor

mel = Tensor(size=(80, 100))
pitch = Tensor(size=(80, 1))

cat = torch.concat([mel, pitch], dim=1)
print(cat.shape)

linear = nn.Linear(101, 100)
x = linear(cat)
print(x.shape)