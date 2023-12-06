import os, logging

import numpy as np

import torch
from torch import nn

from resources.base import Pipeline
from resources.utils import positional_encoding, positional_encoding_by_frame
from model import Codec
from dataset import NSCDataset 

class PM2_pipeline(Pipeline):

    def __init__(self, model: nn.Module, weight_dir: str, log_dir = None, device = None, gpu_id="0", is_speech_resynth=True):
        super().__init__(model, weight_dir, log_dir, device, gpu_id, is_speech_resynth)

    
    def train_step(self, idx, batch):
        mel_input = batch[0].to(self.device) # take only the mel-cepstrum
        pitch = positional_encoding_by_frame(batch[1]).to(self.device)
        mag = batch[2].to(self.device)

        model_output_list, commit_loss_list = self.model(mel_input, pitch, mag)

        # the output from 3 different bitrate=...,
        mel_loss_list = [None] * 3
        
        for i in range(3):
            mel_loss_list[i] = self.criteria(model_output_list[i], mel_input)
            self.train_loss[i] += mel_loss_list[i].item()
            self.train_loss[i+3] += commit_loss_list[i].item()


        if idx % 4000 == 0:
            print('mel loss: {:.4f} {:.4f} {:.4f}; commit loss: {:.4f} {:.4f} {:.4f}'.format(
                    mel_loss_list[0].item(), mel_loss_list[1].item(), mel_loss_list[2].item(),
                    commit_loss_list[0].item(), commit_loss_list[1].item(), commit_loss_list[2].item()))
            
        
        t_l = mel_loss_list[0] + mel_loss_list[1] * 0.2 + mel_loss_list[2] * 0.04 + commit_loss_list[0] + commit_loss_list[1] * 0.2 + commit_loss_list[2] * 0.04
        t_l.backward()

        # gradient = gradient * coeff (< 1)
        nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0)

        self.optimizer.step()

    def val_step(self, idx, batch):
        mel_input = batch[0].to(self.device)
        pitch = positional_encoding_by_frame(batch[1]).to(self.device)
        mag = batch[2].to(self.device)

        model_output_list, commit_loss_list = self.model(mel_input, pitch, mag)

        mel_loss_list = [None] * 3

        for i in range(3):
            mel_loss_list[i] = self.criteria(model_output_list[i], mel_input)
            self.val_loss[i] += mel_loss_list[i].item()

    def log(self):
        logging.error("log?")
    
    def test_step(self):
        logging.error("Not testing")

if __name__ == '__main__':

    root_dir = "/media/daniel0321/LargeFiles/datasets/VCTK"
    model = Codec()
    
    pipeline = PM2_pipeline(
        model = model,
        weight_dir = "models/pitch_and_mag_newencode",
        log_dir = "logger/pitch_and_mag_newencode"
    )

    pipeline.train(
        train_set_path = os.path.join(root_dir, "dataset/train_spc.pkl"),
        test_set_path = os.path.join(root_dir, "dataset/test_spc.pkl") 
    )
