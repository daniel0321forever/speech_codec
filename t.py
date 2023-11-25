import torch
from torch import nn, Tensor

mel = Tensor(size=(1, 100, 80))
pitch = Tensor(size=(1, 1, 80))

cat = torch.concat([mel, pitch], dim=1)
print(cat.shape)

# linear = nn.Linear(101, 100)
# x = linear(cat)
# print(x.shape)

# conv1 = nn.Conv1d(in_channels=100, out_channels=50, kernel_size=1)
# x = conv1(mel)

# print(x.shape)