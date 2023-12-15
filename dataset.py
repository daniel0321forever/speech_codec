import os,re, pickle
import numpy as np
import sys
import librosa
import argparse
import time
import math

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from utils.utils import compute_pitch, compute_mel, positional_encoding

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



class NSCDataset(Dataset):

    def __init__(self):
        self.singer_name_list = []
        self.mel_spec_list = []
        self.linear_spc_list = []
        self.pitch_list = []
        self.mag_list = []


    def add_data(self, audio_dir):
        
        iteration = 1
        for cur_dir in os.listdir(audio_dir):
            singer_name = cur_dir
            cur_singer_dir = os.path.join(audio_dir, cur_dir)
            
            if not os.path.isdir(cur_singer_dir):
                continue
            
            print(f"Singer {iteration} / {len(os.listdir(audio_dir))}")
            iteration += 1

            # print (cur_singer_dir)
            for audio_name in tqdm(os.listdir(cur_singer_dir)):
                audio_path = os.path.join(audio_dir, cur_dir, audio_name)
                
                # print (audio_name, singer_name, audio_path)
                voc, sr = librosa.core.load(audio_path, sr=None, mono=True)

                if sr != 24000:
                    voc = librosa.resample(voc, orig_sr=sr, target_sr=24000)

                
                voc = librosa.util.normalize(voc) * 0.9 # why * 0.9audio_path
                
                voc_mel, voc_spc = compute_mel(voc)

                voc_pitch, voc_mag = compute_pitch(voc)
                max_mag_idx = np.argmax(voc_mag, axis=1)    
                voc_pitch = np.array([[voc_pitch[frame][max_mag_idx[frame]]] for frame in range(len(voc_pitch))]) # (frame, pitch)
                voc_mag = np.array([[voc_mag[frame][max_mag_idx[frame]]] for frame in range(len(voc_mag))]) # (frame mag)
                                
                sample_size = 80

                for i in range(0, len(voc_mel), sample_size):
                    start = i
                    end = i + sample_size
                    end = min(end, len(voc_mel))

                    cur_data = np.array(voc_mel[start:end])
                    # cur_spc = np.array(voc_spc[start:end])
                    cur_pitch = np.array(voc_pitch[start:end])
                    cur_mag = np.array(voc_mag[start:end])

                    if end - start < sample_size:
                        padding_length = sample_size - (end - start)

                        cur_data = np.array(np.pad(cur_data, pad_width=((0, padding_length), (0, 0)), constant_values=-10.0))
                        # cur_spc = np.array(np.pad(cur_spc, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                        cur_pitch = np.array(np.pad(cur_pitch, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                        cur_mag = np.array(np.pad(cur_mag, pad_width=((0, padding_length), (0, 0)), constant_values=0))

                    cur_pitch = positional_encoding(cur_pitch, is_batch=False)
                    cur_mag = positional_encoding(cur_mag, is_batch=False)

                    self.mel_spec_list.append(cur_data)
                    # self.linear_spc_list.append(cur_spc)
                    self.pitch_list.append(cur_pitch)
                    self.mag_list.append(cur_mag)
                    # self.singer_name_list.append(singer_name)

        # self.pitch_list = positional_encoding(self.pitch_list)
        # self.mag_list = positional_encoding(self.mag_list)

    def __len__(self):
        return len(self.mel_spec_list)

    def __getitem__(self, idx):
        # print (self.mel_spec_list[idx].shape)
        return (self.mel_spec_list[idx], self.pitch_list[idx], self.mag_list[idx])
    

class NSCDataset_QPM(Dataset):

    def __init__(self):
        self.singer_name_list = []
        self.mel_spec_list = []
        self.linear_spc_list = []
        self.pitch_list = []
        self.mag_list = []


    def add_data(self, audio_dir):
        
        iteration = 1
        for cur_dir in os.listdir(audio_dir):
            singer_name = cur_dir
            cur_singer_dir = os.path.join(audio_dir, cur_dir)
            
            if not os.path.isdir(cur_singer_dir):
                continue
            
            print(f"Singer {iteration} / {len(os.listdir(audio_dir))}")
            iteration += 1

            # print (cur_singer_dir)
            for audio_name in tqdm(os.listdir(cur_singer_dir)):
                audio_path = os.path.join(audio_dir, cur_dir, audio_name)
                
                # print (audio_name, singer_name, audio_path)
                voc, sr = librosa.core.load(audio_path, sr=None, mono=True)

                if sr != 24000:
                    voc = librosa.resample(voc, orig_sr=sr, target_sr=24000)

                
                voc = librosa.util.normalize(voc) * 0.9 # why * 0.9audio_path
                
                voc_mel, voc_spc = compute_mel(voc)

                voc_pitch, voc_mag = compute_pitch(voc)
                max_mag_idx = np.argmax(voc_mag, axis=1)    
                voc_pitch = np.array([[max_mag_idx[frame]] for frame in range(len(voc_pitch))]) # (frames, 1)
                voc_mag = np.array([[voc_mag[frame][max_mag_idx[frame]]] for frame in range(len(voc_mag))]) # (frames, 1)
                                
                sample_size = 80

                for i in range(0, len(voc_mel), sample_size):
                    start = i
                    end = i + sample_size
                    end = min(end, len(voc_mel))

                    cur_data = np.array(voc_mel[start:end])
                    # cur_spc = np.array(voc_spc[start:end])
                    cur_pitch = np.array(voc_pitch[start:end])
                    cur_mag = np.array(voc_mag[start:end])

                    if end - start < sample_size:
                        padding_length = sample_size - (end - start)

                        cur_data = np.array(np.pad(cur_data, pad_width=((0, padding_length), (0, 0)), constant_values=-10.0))
                        # cur_spc = np.array(np.pad(cur_spc, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                        cur_pitch = np.array(np.pad(cur_pitch, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                        cur_mag = np.array(np.pad(cur_mag, pad_width=((0, padding_length), (0, 0)), constant_values=0))

                    self.mel_spec_list.append(cur_data)
                    # self.linear_spc_list.append(cur_spc)
                    self.pitch_list.append(cur_pitch)
                    self.mag_list.append(cur_mag)
                    # self.singer_name_list.append(singer_name)

    def __len__(self):
        return len(self.mel_spec_list)

    def __getitem__(self, idx):
        # print (self.mel_spec_list[idx].shape)
        return (self.mel_spec_list[idx], self.pitch_list[idx], self.mag_list[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', required=False, default="/media/daniel0321/LargeFiles/datasets/VCTK/raw_data/train/wav48_silence_trimmed", type=str)
    parser.add_argument('-o', '--output', required=False, default="/media/daniel0321/LargeFiles/datasets/VCTK/dataset/train_spc.pkl", type=str)
    
    args = parser.parse_args()
    dataset_dir = args.datadir
    output_path = args.output

    print(dataset_dir)
    print(output_path)

    cur_dataset = NSCDataset_QPM()
    cur_dataset.add_data(dataset_dir)
    # cur_dataset.add_data(dataset_dir[1])

    with open(output_path, 'wb') as f:
        pickle.dump(cur_dataset, f)

    print("done")
    print ("Number of data:", len(cur_dataset))
    # print (cur_dataset[0])
    print (cur_dataset[0][0].shape, cur_dataset[0][1].shape, cur_dataset[0][2].shape) # mel_spec, linear_spc
