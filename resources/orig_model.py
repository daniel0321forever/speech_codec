import os,re
import numpy as np
import sys
import librosa
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from vector_quantize_pytorch import VectorQuantize

sys.path.append(os.path.join(os.path.dirname(__file__), '../AutoSVS'))
from parallel_wavegan.utils import load_model



class MelResCodec(nn.Module):
    def __init__(self, device='cpu'):
        super(MelResCodec, self).__init__()
        
        self.device = device
        self.input_dim = 513 + 80
        self.latent_dim = 256
        self.output_dim = 513

        self.encoder_linear = nn.Sequential(
                nn.Conv1d(self.input_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
            )

        self.res_encoder_block = ResBlock(self.latent_dim)
        
        self.vq = VectorQuantize(
                        dim = self.latent_dim,
                        codebook_size = 256,     
                        decay = 0.99,      
                        commitment = 1.
                    )
        
        self.res_decoder_block = ResBlock(self.latent_dim)
        self.decoder_linear = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.output_dim, 1, stride=1, padding=0, dilation=1)
            )

    def forward(self, x):

        x = x.contiguous().transpose(1, 2)
        x = self.encoder_linear(x)

        x1 = self.res_encoder_block(x)
        x1_quantized, x1_indices, x1_commit_loss = self.vq(x1.transpose(1, 2))
        x1_quantized = x1_quantized.transpose(1, 2)

        x1_decoded = self.res_decoder_block(x1_quantized)
        x1_decoded_output = self.decoder_linear(x1_decoded)
        x1_decoded_output = x1_decoded_output.contiguous().transpose(1, 2)

        return x1_decoded_output, x1_commit_loss

class ResBlock(nn.Module):
    ''' A two-feed-forward-layer module with no transpose (should be performed beforehand) '''

    def __init__(self, latent_dim, temp_downsample_rate=1):
        super(ResBlock, self).__init__()
        self.latent_dim = latent_dim
        self.kernel_size = 3
        self.padding = (self.kernel_size - 1) // 2
        self.temp_downsample_rate = temp_downsample_rate

        self.res1 = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=1, padding=self.padding, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=1, padding=self.padding, dilation=1),
        )

        self.res2 = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=1, padding=self.padding, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=1, padding=self.padding, dilation=1),
        )

        self.output_linear = nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1)

        if self.temp_downsample_rate == 2:
            self.ds = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=2, padding=self.padding, dilation=1),
            )

        elif self.temp_downsample_rate == 4:
            self.ds = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=2, padding=self.padding, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=2, padding=self.padding, dilation=1),
            )

    def forward(self, x):
        out = self.res1(x)
        x = x + out

        out = self.res2(x)
        x = x + out

        x = self.output_linear(x)

        if self.temp_downsample_rate == 2:
            ds_x = self.ds(x)
            ds_x = torch.repeat_interleave(ds_x, 2, dim=2)
            return ds_x, x

        elif self.temp_downsample_rate == 4:
            ds_x = self.ds(x)
            ds_x = torch.repeat_interleave(ds_x, 4, dim=2)
            return ds_x, x

        else:
            return x

class Codec(nn.Module):
    def __init__(self, device='cpu'):
        super(Codec, self).__init__()
        
        self.device = device
        # self.input_dim = 513
        self.input_dim = 80
        self.latent_dim = 256

        self.encoder_linear = nn.Sequential(
                nn.Conv1d(self.input_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
            )

        self.res_encoder_block1 = ResBlock(self.latent_dim, temp_downsample_rate=4)
        self.res_encoder_block2 = ResBlock(self.latent_dim, temp_downsample_rate=2)
        self.res_encoder_block3 = ResBlock(self.latent_dim)

        self.vq1 = VectorQuantize(
                        dim = self.latent_dim,
                        codebook_size = 256,     
                        decay = 0.99,      
                        commitment = 1.
                    )

        self.vq2 = VectorQuantize(
                        dim = self.latent_dim,
                        codebook_size = 256,     
                        decay = 0.99,      
                        commitment = 1.
                    )

        self.vq3 = VectorQuantize(
                        dim = self.latent_dim,
                        codebook_size = 256,     
                        decay = 0.99,      
                        commitment = 1.
                    )
        
        self.res_decoder_block1 = ResBlock(self.latent_dim)
        self.res_decoder_block2 = ResBlock(self.latent_dim)
        self.res_decoder_block3 = ResBlock(self.latent_dim)
        self.decoder_linear = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.input_dim, 1, stride=1, padding=0, dilation=1)
            )

    def encode(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.encoder_linear(x)

        # First CNN + VQ compressor
        ds_x1, x1 = self.res_encoder_block1(x)
        x1_quantized, x1_indices, x1_commit_loss = self.vq1(ds_x1.transpose(1, 2))
        x1_quantized = x1_quantized.transpose(1, 2)
        x1_quantize_residual = x1 - x1_quantized

        ds_x2, x2 = self.res_encoder_block2(x1_quantize_residual)
        x2_quantized, x2_indices, x2_commit_loss = self.vq2(ds_x2.transpose(1, 2))
        x2_quantized = x2_quantized.transpose(1, 2)
        x2_quantize_residual = x2 - x2_quantized

        x3 = self.res_encoder_block3(x2_quantize_residual)
        x3_quantized, x3_indices, x3_commit_loss = self.vq3(x3.transpose(1, 2))
        x3_quantized = x3_quantized.transpose(1, 2)

        return x1_indices, x2_indices, x3_indices

    def forward(self, x):

        x = x.contiguous().transpose(1, 2) # (bins=80,frames=80)
        x = self.encoder_linear(x) # (frames, feats)

        # First CNN + VQ compressor
        ds_x1, x1 = self.res_encoder_block1(x)
        x1_quantized, x1_indices, x1_commit_loss = self.vq1(ds_x1.transpose(1, 2)) # (frames, features)
        x1_quantized = x1_quantized.transpose(1, 2) # (feats, frames)
        x1_quantize_residual = x1 - x1_quantized

        ds_x2, x2 = self.res_encoder_block2(x1_quantize_residual)
        x2_quantized, x2_indices, x2_commit_loss = self.vq2(ds_x2.transpose(1, 2))
        x2_quantized = x2_quantized.transpose(1, 2)
        x2_quantize_residual = x2 - x2_quantized

        x3 = self.res_encoder_block3(x2_quantize_residual)
        x3_quantized, x3_indices, x3_commit_loss = self.vq3(x3.transpose(1, 2))
        x3_quantized = x3_quantized.transpose(1, 2)

        x1_decoded = self.res_decoder_block3(x1_quantized) # (feats, frames)

        x2_decoded = self.res_decoder_block2(x2_quantized)
        x2_decoded = self.res_decoder_block3(x2_decoded + x1_quantized)

        x3_decoded = self.res_decoder_block1(x3_quantized)
        x3_decoded = self.res_decoder_block2(x3_decoded + x2_quantized)
        x3_decoded = self.res_decoder_block3(x3_decoded + x1_quantized)
        

        x1_decoded_output = self.decoder_linear(x1_decoded)
        x1_decoded_output = x1_decoded_output.contiguous().transpose(1, 2) # (frames, bins)

        x2_decoded_output = self.decoder_linear(x2_decoded)
        x2_decoded_output = x2_decoded_output.contiguous().transpose(1, 2)

        x3_decoded_output = self.decoder_linear(x3_decoded)
        x3_decoded_output = x3_decoded_output.contiguous().transpose(1, 2)
        
        return (x1_decoded_output, x2_decoded_output, x3_decoded_output), (x1_commit_loss, x2_commit_loss, x3_commit_loss)