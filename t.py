import pickle
from dataset import NSCDataset
from torch.utils.data import DataLoader

with open('dataset/train_spc.pkl', 'rb') as f:
        train_dataset = pickle.load(f)

loader = DataLoader(train_dataset, batch_size=1)

for i, batch in enumerate(loader):
    if i > 0:
          break
    
    mel, linear, pitch, mag = batch

    print(mel.shape)
    print(linear.shape)
    print(pitch.shape)
    print(mag.shape)