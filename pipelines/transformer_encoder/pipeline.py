import os, logging, sys

import numpy as np

import torch
from torch import nn
from torch.nn.modules import Module

from utils.base import Pipeline
from utils.utils import compute_mel, compute_pitch
from dataset import NSCDataset
from model import CodecT


class TransformerCodecPipeline(Pipeline):

    def __init__(self, model: nn.Module, weight_dir: str, log_dir: str | None, device: str | None = None, gpu_id="0", is_speech_resynth=True, sr=24000):
        super().__init__(model, weight_dir, log_dir, device, gpu_id, is_speech_resynth, sr)

        self.loss_len = 6 # total kinds of loss recorded on tensorboard
    
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
        
        for i in range(self.loss_len - 3):
            self.train_loss[i+3] += commit_loss_list[i].item()


        if idx % 4000 == 0:
            print("mel loss: ", end="")
            for i in range(3):
                print(mel_loss_list[i].item(), end=" ")
            print("commit loss: ", end="")
            for i in range(self.loss_len - 3):
                print(commit_loss_list[i].item(), end=" ")
        
        t_l = mel_loss_list[0] + mel_loss_list[1] * 0.2 + mel_loss_list[2] * 0.04 + commit_loss_list[0] + commit_loss_list[1] * 0.2 + commit_loss_list[2] * 0.04
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
        sample_size = 80
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

                pitch_input = torch.tensor(pitch_segments[i]).unsqueeze(0).to(self.device)
                mag_input = torch.tensor(mag_segments[i]).unsqueeze(0).to(self.device)
                
                x1_quantized, x2_quantized, x3_quantized = model.encoder(model_input, pitch_input, mag_input)
                bos = model.bos

                combined_input = bos
                x1_residual_input = bos
                x2_residual_input = bos
                for frame in range(x1_quantized.shape[1]):
                    model_output_list = model.decoder(
                        combined_input=combined_input,
                        x1_quantize_residual_input=x1_residual_input,
                        x2_quantize_residual_input=x2_residual_input,
                    )

                    new_combined =  model_output_list[3]
                    new_x1_res = model_output_list[4] - model_output_list[3]
                    new_x2_res = model_output_list[5] - model_output_list[4]

                    combined_input = torch.concat([combined_input, new_combined[:, -1, :]], dim=1)
                    x1_residual_input = torch.concat([x1_residual_input, new_x1_res[:, -1, :]], dim=1)
                    x2_residual_input = torch.concat([x2_residual_input, new_x2_res[:, -1, :]], dim=1)
                
                print(model_output_list[0].shape)

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


if __name__ == '__main__':

    data_root_dir = "/media/daniel0321/LargeFiles/datasets/VCTK"
    pipeline_dir = "."
    model = CodecT()

    pipeline = TransformerCodecPipeline(
        model=model,
        weight_dir="models",
        log_dir = "logger",
    )

    # pipeline.train(
    #     train_set_path = os.path.join(data_root_dir, "dataset/train_spc.pkl"),
    #     test_set_path = os.path.join(data_root_dir, "dataset/test_spc.pkl"),
    #     batch_size=16,
    # )

    pipeline.test("/media/daniel0321/LargeFiles/datasets/VCTK/raw_data/test/wav48_silence_trimmed/")

    pipeline.generate(audio_path="../../raw_data_sample/test/wav48_silence_trimmed/p343/p343_004_mic1.flac", save_dir="performance")