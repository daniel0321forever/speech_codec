import os, logging, sys

import numpy as np

import torch
from torch import nn
from torch.nn.modules import Module

from utils.base import Pipeline
from utils.utils import compute_mel, compute_pitch, PositionalEncoding
from dataset import NSCDataset
from model import CodecT


class TransformerCodecPipeline(Pipeline):

    def __init__(self, model: nn.Module, weight_dir: str, log_dir: str | None, device: str | None = None, gpu_id="0", is_speech_resynth=True, sr=24000):
        super().__init__(model, weight_dir, log_dir, device, gpu_id, is_speech_resynth, sr)

        self.positional_encoder = PositionalEncoding()
        self.loss_len = 2
    
    def train_step(self, idx, batch):

        mel_input = batch[0].to(self.device) # take only the mel-cepstrum
        pitch = batch[1].to(self.device)
        pitch = self.positional_encoder(pitch)

        model_output, commit_loss = self.model(mel_input, pitch)

        # find loss
        mel_loss = self.criteria(model_output, mel_input)
        self.train_loss[0] += mel_loss.item()
        self.train_loss[1] += commit_loss.item()

        if idx % 4000 == 0:
            print("\nmel loss: {:.4f}".format(mel_loss.item()))
            print("commit loss: {:.4f}".format(commit_loss.item()))
        
        # ! modified
        t_l = mel_loss + commit_loss
        t_l.backward()

        # gradient = gradient * coeff (< 1)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

    
    def val_step(self, idx, batch):

        mel_input = batch[0].to(self.device)
        pitch = batch[1].to(self.device)
        pitch = self.positional_encoder(pitch)

        model_output, commit_loss = self.model(mel_input, pitch)

        mel_loss = self.criteria(model_output,  mel_input)
        self.val_loss[0] += mel_loss.item()

    def inference(self, source_path: str):

        """
        Why sould we reprocess the data?
        """
        
        # get input
        y = self.source_y
        mel = self.source_mel_feature
        pitch = compute_pitch(y)

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

                if end - start < sample_size:
                    padding_length = sample_size - (end - start)
                    
                    cur_data = np.array(np.pad(cur_data, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                    cur_pitch = np.array(np.pad(cur_pitch, pad_width=(padding_length, 0), constant_values=0))

                mel_segments.append(cur_data)
                pitch_segments.append(cur_pitch)
                
        pred_list_vq = []

        with torch.no_grad():
            for i in range(len(mel_segments)):
                model_input = torch.tensor(mel_segments[i]).unsqueeze(0).to(self.device)
                pitch_input = torch.Tensor(pitch_segments[i]).unsqueeze(0).to(self.device)
                pitch_input = self.positional_encoder(pitch_input)
                
                x_quantized = model.encoder(model_input)
                bos = model.bos[:1]

                mel_input = bos.to(self.device) # TODO: The mel_input for decoder is the right-shifted output (dim = 80), so that we can use decoder output to do auto-regressive 

                for frame in range(x_quantized.shape[1]):
                    mel_output = model.decoder(
                        pitch_input=pitch_input[:, :frame+1, :],
                        mel_input=mel_input,
                        x_quantized=x_quantized, 
                    )
                                        
                    mel_input = torch.concat([mel_input, mel_output[:, -1:, :]], dim=1)
                
                # TODO: remember, the output from the previous process is combined
                waveform_output_vq, _ = self.vocoder(mel_output, output_all=True)
                pred_list_vq.append(waveform_output_vq[0].cpu())

        output_wav = np.concatenate(pred_list_vq, axis=0)

        return output_wav


if __name__ == '__main__':

    data_root_dir = "/media/daniel0321/LargeFiles/datasets/VCTK/dataset"
    pipeline_dir = "."

    trainset_path =  os.path.join(data_root_dir, "train_spc.pkl")
    testset_path = os.path.join(data_root_dir, "test_spc.pkl")
    testset_dir = "/media/daniel0321/LargeFiles/datasets/VCTK/raw_data/test/wav48_silence_trimmed"

    model = CodecT(
        device="cuda"
    )

    pipeline = TransformerCodecPipeline(
        model=model,
        weight_dir="models",
        log_dir = "logger",
    )

    pipeline.train(
        train_set_path = trainset_path,
        test_set_path = testset_path,
        batch_size=16,
    )

    pipeline.test(test_dir=testset_dir)

    pipeline.generate(audio_path="../../raw_data_sample/test/wav48_silence_trimmed/p343/p343_004_mic1.flac", save_dir="performance")