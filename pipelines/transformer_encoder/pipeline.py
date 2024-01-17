import os, logging, sys

import numpy as np

import torch
from torch import nn
from torch.nn.modules import Module

from utils.base import Pipeline
from utils.utils import positional_encoding, positional_encoding_by_frame, compute_mel, compute_pitch
from dataset import NSCDataset
from model import Codec


class PM_QUANTIZED(Pipeline):

    def __init__(self, model: nn.Module, weight_dir: str, log_dir: str | None, device: str | None = None, gpu_id="0", is_speech_resynth=True, sr=24000):
        super().__init__(model, weight_dir, log_dir, device, gpu_id, is_speech_resynth, sr)

        self.loss_len = 8
    
    def train_step(self, idx, batch):

        mel_input = batch[0].to(self.device) # take only the mel-cepstrum
        pitch = batch[1].to(self.device)
        mag = batch[2].to(self.device)

        model_output_list, commit_loss_list = self.model(mel_input, pitch, mag)

        # the output from 3 different bitrate=...,
        mel_loss_list = [None] * 3
        
        for i in range(3):
            mel_loss_list[i] = self.criteria(model_output_list[i], mel_input)
            self.train_loss[i] += mel_loss_list[i].item()
        
        for i in range(5):
            self.train_loss[i+3] += commit_loss_list[i].item()


        if idx % 4000 == 0:
            print('mel loss: {:.4f} {:.4f} {:.4f}; commit loss: {:.4f} {:.4f} {:.4f}'.format(
                    mel_loss_list[0].item(), mel_loss_list[1].item(), mel_loss_list[2].item(),
                    commit_loss_list[0].item(), commit_loss_list[1].item(), commit_loss_list[2].item()), commit_loss_list[3].item(), commit_loss_list[4].item())
            
        
        t_l = mel_loss_list[0] + mel_loss_list[1] * 0.2 + mel_loss_list[2] * 0.04 + commit_loss_list[0] + commit_loss_list[1] * 0.2 + commit_loss_list[2] * 0.04 + commit_loss_list[3] * 8 + commit_loss_list[4] * 8
        t_l.backward()

        # gradient = gradient * coeff (< 1)
        nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0)

        self.optimizer.step()

    
    def val_step(self, idx, batch):

        mel_input = batch[0].to(self.device)
        pitch = batch[1].to(self.device)
        mag = batch[2].to(self.device)

        model_output_list, commit_loss_list = self.model(mel_input, pitch, mag)

        mel_loss_list = [None] * 3

        for i in range(3):
            mel_loss_list[i] = self.criteria(model_output_list[i], mel_input)
            self.val_loss[i] += mel_loss_list[i].item()

    def inference(self, source_path: str):
        
        # get input
        y = self.source_y
        mel = self.source_mel_feature

        pitch, mag = compute_pitch(y)
        max_mag_idx = np.argmax(mag, axis=1)    
        pitch = np.array([[pitch[frame][max_mag_idx[frame]]] for frame in range(len(pitch))])
        mag = np.array([[mag[frame][max_mag_idx[frame]]] for frame in range(len(mag))])

        # slice data
        sample_size = 200
        mel_segments = []
        pitch_segments = []
        mag_segments = []

        for i in range(0, len(mel), sample_size):
            start = i
            end = min(i+sample_size, len(mel))
            
            if start + 1 < end:
                cur_data = np.array(mel[start:end])
                cur_pitch = np.array(pitch[start:end])
                cur_mag = np.array(mag[start:end])

                if end - start < sample_size:
                    padding_length = sample_size - (end - start)
                    
                    cur_data = np.array(np.pad(cur_data, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                    cur_pitch = np.array(np.pad(cur_pitch, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                    cur_mag = np.array(np.pad(cur_mag, pad_width=((0, padding_length), (0, 0)), constant_values=0))

                mel_segments.append(cur_data)
                pitch_segments.append(cur_pitch)
                mag_segments.append(cur_mag)
                
        pred_list_vq1 = []
        pred_list_vq2 = []
        pred_list_vq3 = []

        with torch.no_grad():
            for i in range(len(mel_segments)):
                model_input = torch.tensor(mel_segments[i]).unsqueeze(0).to(self.device)
                pitch_input = positional_encoding(torch.tensor(pitch_segments[i]).unsqueeze(0)).to(self.device)
                mag_input = positional_encoding(torch.tensor(mag_segments[i]).unsqueeze(0)).to(self.device)
                
                model_input = torch.log(model_input + 1e-10)
                
                model_output_list, commit_loss_list = model(model_input, pitch_input, mag_input)

                waveform_output_vq1, _ = self.vocoder(model_output_list[0], output_all=True)
                waveform_output_vq2, _ = self.vocoder(model_output_list[1], output_all=True)
                waveform_output_vq3, _ = self.vocoder(model_output_list[2], output_all=True)
                pred_list_vq1.append(waveform_output_vq1[0].cpu())
                pred_list_vq2.append(waveform_output_vq2[0].cpu())
                pred_list_vq3.append(waveform_output_vq3[0].cpu())

        output_wav1 = np.concatenate(pred_list_vq1, axis=0)
        output_wav2 = np.concatenate(pred_list_vq2, axis=0)
        output_wav3 = np.concatenate(pred_list_vq3, axis=0)

        return output_wav1, output_wav2, output_wav3


class PM_ENcodec(PM_QUANTIZED):

    def __init__(self, model: nn.Module, weight_dir: str, log_dir: str | None, device: str | None = None, gpu_id="0", is_speech_resynth=True, sr=24000):
        super().__init__(model, weight_dir, log_dir, device, gpu_id, is_speech_resynth, sr)

if __name__ == '__main__':

    data_root_dir = "/media/daniel0321/LargeFiles/datasets/VCTK"
    pipeline_dir = "."
    model = Codec()

    pipeline = PM_QUANTIZED(
        model=model,
        weight_dir="models",
        log_dir = "logger",
    )

    pipeline.train(
        train_set_path = os.path.join(data_root_dir, "dataset/train_spc.pkl"),
        test_set_path = os.path.join(data_root_dir, "dataset/test_spc.pkl"),
        batch_size=16,
    )

    pipeline.test("/media/daniel0321/LargeFiles/datasets/VCTK/raw_data/test/wav48_silence_trimmed/")

    pipeline.generate(audio_path="../../raw_data_sample/test/wav48_silence_trimmed/p343/p343_004_mic1.flac", save_dir="performance")