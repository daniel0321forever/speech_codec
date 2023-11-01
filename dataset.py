import os,re, pickle
import numpy as np
import sys
import librosa
import argparse
import time
import math

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

sample_rate = 24000
fft_size = 2048
win_length = 1200
hop_length = 300
# win_length = 2048
# hop_length = 120

# num_mels = 80
# fmin = 0
# fmax = sample_rate / 2

supp_fft_size = 1024
supp_hop_length = 300

num_mels = 80
fmin = 80
fmax = 7600

def compute_mel(y):
    
    eps = 1e-10
    log_base = 10.0

    x_stft = librosa.stft(
        y,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))


    x_stft_supp = librosa.stft(
        y,
        n_fft=supp_fft_size,
        hop_length=supp_hop_length,
        win_length=supp_fft_size,
        center=True,
        pad_mode="reflect",
    )
    spc_supp = np.abs(x_stft_supp).T

    # Get log spec
    # spc = librosa.amplitude_to_db(spc)

    # print (mel.shape)
    return np.log10(mel), spc_supp


class NSCDataset(Dataset):

    def __init__(self):
        self.singer_name_list = []
        self.mel_spec_list = []
        self.linear_spc_list = []

    def add_data(self, audio_dir):
        for cur_dir in os.listdir(audio_dir):
            singer_name = cur_dir
            cur_singer_dir = os.path.join(audio_dir, cur_dir)
            
            if not os.path.isdir(cur_singer_dir):
                continue

            print (cur_singer_dir)
            for audio_name in tqdm(os.listdir(cur_singer_dir)):
                audio_path = os.path.join(audio_dir, cur_dir, audio_name)
                
                # print (audio_name, singer_name, audio_path)
                voc, sr = librosa.core.load(audio_path, sr=None, mono=True)

                if sr != 24000:
                    voc = librosa.resample(voc, orig_sr=sr, target_sr=24000)

                voc = librosa.util.normalize(voc) * 0.9 # why * 0.9
                voc_mel, voc_spc = compute_mel(voc)
                # print (voc_mel.shape)
                sample_size = 80

                for i in range(0, len(voc_mel), sample_size):
                    start = i
                    end = i + sample_size
                    end = min(end, len(voc_mel))

                    cur_data = np.array(voc_mel[start:end])
                    cur_spc = np.array(voc_spc[start:end])

                    if end - start < sample_size:
                        padding_length = sample_size - (end - start)
                        # print (cur_data.shape)
                        cur_data = np.array(np.pad(cur_data, pad_width=((0, padding_length), (0, 0)), constant_values=-10.0))
                        cur_spc = np.array(np.pad(cur_spc, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                        # print (cur_data.shape)
                    self.mel_spec_list.append(cur_data)
                    self.linear_spc_list.append(cur_spc)
                    self.singer_name_list.append(singer_name)

    def __len__(self):
        return len(self.mel_spec_list)

    def __getitem__(self, idx):
        # print (self.mel_spec_list[idx].shape)
        return (self.mel_spec_list[idx], self.linear_spc_list[idx])

if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    output_path = sys.argv[2]

    cur_dataset = NSCDataset()
    cur_dataset.add_data(dataset_dir)
    # cur_dataset.add_data(dataset_dir[1])

    with open(output_path, 'wb') as f:
        pickle.dump(cur_dataset, f)

    print (len(cur_dataset))
    print (cur_dataset[0])
    print (cur_dataset[0][0].shape, cur_dataset[0][1].shape)