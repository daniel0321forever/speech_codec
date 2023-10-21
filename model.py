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

import h5py
def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data

class PWGVocoder(nn.Module):
    def __init__(self, device, normalize_path, vocoder_path):
        super(PWGVocoder, self).__init__()
        
        self.device = device
        self.normalize_mean = torch.tensor(read_hdf5(normalize_path, "mean")).to(self.device)
        self.normalize_scale = torch.tensor(read_hdf5(normalize_path, "scale")).to(self.device)

        self.vocoder = load_model(vocoder_path)
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.device)


        # stat_data = np.load(normalize_path)

        # self.normalize_mean = torch.tensor(stat_data[0]).to(self.device)
        # self.normalize_scale = torch.tensor(stat_data[1]).to(self.device)

        # Freeze vocoder weight
        for p in self.vocoder.parameters():
            p.requires_grad = False

        self.max_vocoder_segment_length = 400

    def forward(self, spec_output, output_all=False):
        # Go through the vocoder to generate waveform
        spec_output_norm = (spec_output - self.normalize_mean) / self.normalize_scale
        # print (self.normalize_mean, self.normalize_scale)

        # Pick at most "self.max_vocoder_segment_length" frames, in order to avoid CUDA OOM.
        # x is the random noise for vocoder
        if spec_output_norm.shape[1] > self.max_vocoder_segment_length and output_all == False:
            start_frame = int(torch.rand(1) * (spec_output_norm.shape[1] - self.max_vocoder_segment_length))
            end_frame = start_frame + self.max_vocoder_segment_length
            spec_for_vocoder = torch.nn.ReplicationPad1d(2)(spec_output_norm[:,start_frame:end_frame,:].transpose(1, 2))
            x = torch.randn(spec_output_norm.shape[0], 1, self.max_vocoder_segment_length * self.vocoder.upsample_factor).to(self.device)
        else:
            start_frame = 0
            spec_for_vocoder = torch.nn.ReplicationPad1d(2)(spec_output_norm.transpose(1, 2))
            x = torch.randn(spec_output_norm.shape[0], 1, spec_output_norm.shape[1] * self.vocoder.upsample_factor).to(self.device)
        
        # print (x.shape, spec_output_norm.transpose(1, 2).shape)
        waveform_output = self.vocoder(x, spec_for_vocoder).squeeze(1)

        return waveform_output, start_frame

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
        # transmit the compressed values (code book index of each input vector) to decoder
        # transmit the quantized tensor (decoded tensor) to the next level

        x = x.contiguous().transpose(1, 2)
        x = self.encoder_linear(x)

        # First CNN + VQ compressor
        ds_x1, x1 = self.res_encoder_block1(x)
        # quantized vectors, code book index of each vector, the loss between quantized tensor and tensor
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

        # decode x1
        x1_decoded = self.res_decoder_block3(x1_quantized)

        # decode x2
        x2_decoded = self.res_decoder_block2(x2_quantized)
        x2_decoded = self.res_decoder_block3(x2_decoded + x1_quantized)

        # decode x3
        x3_decoded = self.res_decoder_block1(x3_quantized)
        x3_decoded = self.res_decoder_block2(x3_decoded + x2_quantized)
        x3_decoded = self.res_decoder_block3(x3_decoded + x1_quantized)
        

        x1_decoded_output = self.decoder_linear(x1_decoded)
        x1_decoded_output = x1_decoded_output.contiguous().transpose(1, 2)

        x2_decoded_output = self.decoder_linear(x2_decoded)
        x2_decoded_output = x2_decoded_output.contiguous().transpose(1, 2)

        x3_decoded_output = self.decoder_linear(x3_decoded)
        x3_decoded_output = x3_decoded_output.contiguous().transpose(1, 2)
        
        return (x1_decoded_output, x2_decoded_output, x3_decoded_output), (x1_commit_loss, x2_commit_loss, x3_commit_loss)
