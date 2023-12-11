import os,re
import numpy as np
import sys
import librosa
import argparse
import time
import math
import logging

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
            # 取「結尾減去max_length長度」之時間點之前的一點作為開始
            start_frame = int(torch.rand(1) * (spec_output_norm.shape[1] - self.max_vocoder_segment_length))
            # 取該時間點之後加上max_length作為終點
            end_frame = start_frame + self.max_vocoder_segment_length
            # Pads the input tensor using replication of the input boundary.
            spec_for_vocoder = torch.nn.ReplicationPad1d(2)(spec_output_norm[:,start_frame:end_frame,:].transpose(1, 2)) # time is the last dimension
            # x is the random noise for vocoder [spec output shape 0, 1, trimmed spec output shape 1 * unsample factor]
            x = torch.randn(spec_output_norm.shape[0], 1, self.max_vocoder_segment_length * self.vocoder.upsample_factor).to(self.device)
        else:
            start_frame = 0
            # Pads the input tensor using replication of the input boundary.
            spec_for_vocoder = torch.nn.ReplicationPad1d(2)(spec_output_norm.transpose(1, 2))
            # x is the random noise for vocoder
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
                        commitment_weight = 1.
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

        # downsample by using stride = 2 convolution once
        if self.temp_downsample_rate == 2:
            self.ds = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=2, padding=self.padding, dilation=1),
            )

        # downsample by using stride = 2 convolution twice
        elif self.temp_downsample_rate == 4:
            self.ds = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=2, padding=self.padding, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=2, padding=self.padding, dilation=1),
            )

    
    def forward(self, x):
        """
        x -> res_1 -> res_2 -> output_linear -> downsample -> repeat downsample element
        """

        out = self.res1(x)
        x = x + out

        out = self.res2(x)
        x = x + out

        x = self.output_linear(x)

        # repeat the downsampled elements so that the length is the same
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
        self.input_dim = 80 # Yeah, it is the frequencies dimension
        self.latent_dim = 256
        self.frames = 80
        self.pitch_dim = 64

        self.encoder_linear = nn.Sequential(
                nn.Conv1d(self.input_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
            )
        

        # input (bins=pitch_dim, frames) -> output (bins=pitch_dims, frames)
        self.pitch_encoder = nn.Sequential(
                nn.Conv1d(self.pitch_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
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
                        commitment_weight = 1.
                    )

        self.vq2 = VectorQuantize(
                        dim = self.latent_dim,
                        codebook_size = 256,     
                        decay = 0.99,      
                        commitment_weight = 1.
                    )

        self.vq3 = VectorQuantize(
                        dim = self.latent_dim,
                        codebook_size = 256,     
                        decay = 0.99,      
                        commitment_weight = 1.
                    )
        
        self.vq_pitch = VectorQuantize(
                        dim = self.latent_dim, # positional encoding dim
                        codebook_size = 128,
                        decay = 0.99,
                        commitment_weight = 1,
                    )

        self.res_decoder_block1 = ResBlock(self.latent_dim)
        self.res_decoder_block2 = ResBlock(self.latent_dim)
        self.res_decoder_block3 = ResBlock(self.latent_dim)

        # self.mag_decoder = nn.Sequential(
        #     nn.Conv1d(1, 1, 3, stride=1, padding=1, dilation=1),
        #     nn.LeakyReLU(),
        #     nn.Conv1d(1, 1, 3, stride=1, padding=1, dilation=1)        
        # )

        self.fully_connected1 = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
        )
        self.fully_connected2 = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
        )
        self.fully_connected3 = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
        )

        self.decoder_linear = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.input_dim, 1, stride=1, padding=0, dilation=1)
            )
    
    def encode(self, x, pitch, mag):
        # transmit the compressed values (code book index of each input vector) to decoder
        # transmit the quantized tensor (decoded tensor) to the next level

        x = x.contiguous().transpose(1, 2) # (freq, frames)
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

        # decode pitch, mag
        pitch = self.pitch_encoder(pitch.transpose(1, 2)) # (64, 80)
        pitch_quantized, pitch_indices, pitch_commit_loss = self.vq_pitch(pitch)
        pitch_quantized = pitch_quantized.transpose(1, 2) # (80, 64)
        # mag = self.mag_decoder(mag.transpose(1, 2)) # (64, 80)

        return x1_indices, x2_indices, x3_indices, pitch_indices

    def forward(self, x: torch.Tensor, pitch: torch.Tensor):

        x = x.contiguous().transpose(1, 2) # tranpose the input into (bins, frames)
        x = self.encoder_linear(x)
        
        pitch = pitch.contiguous().transpose(1, 2) # transpose the input into (encoded features = 64, frames)

        # pitch quantize
        pitch = self.pitch_encoder(pitch)
        pitch_quantized, pitch_indices, pitch_commit_loss = self.vq_pitch(pitch.transpose(1, 2)) # (frames=80, features=256)
        pitch_quantized = pitch_quantized.transpose(1, 2)

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
        x1_decoded = torch.concat([x1_decoded, pitch_quantized], dim=1)
        x1_decoded = x1_decoded.contiguous().transpose(1, 2)
        x1_decoded = self.fully_connected3(x1_decoded)

        # decode x2:
        #   decode x2_quantized (residual of x1) -> decode (x2_decoded + x1_quantized)
        x2_decoded = self.res_decoder_block2(x2_quantized)
        x2_decoded = self.res_decoder_block3(x2_decoded + x1_quantized) # TODO: Change to x1_decoded
        x2_decoded = torch.concat([x2_decoded, pitch_quantized], dim=1)
        x2_decoded = x2_decoded.contiguous().transpose(1, 2)
        x2_decoded = self.fully_connected2(x2_decoded)

        # decode x3 (The flow written in the report)
        #   decode x3_quantized -> decode x3_decoded + x2_quantized -> decode x3x2_decoded + x1_quantized
        x3_decoded = self.res_decoder_block1(x3_quantized)
        x3_decoded = self.res_decoder_block2(x3_decoded + x2_quantized) # TODO: Change to x2_decoded
        x3_decoded = self.res_decoder_block3(x3_decoded + x1_quantized) # TODO: Change to x1_decoded
        x3_decoded = torch.concat([x3_decoded, pitch_quantized], dim=1)
        x3_decoded = x3_decoded.contiguous().transpose(1, 2)
        x3_decoded = self.fully_connected1(x3_decoded)
        

        x1_decoded_output = self.decoder_linear(x1_decoded.transpose(1, 2))
        x1_decoded_output = x1_decoded_output.contiguous().transpose(1, 2)

        x2_decoded_output = self.decoder_linear(x2_decoded.transpose(1, 2))
        x2_decoded_output = x2_decoded_output.contiguous().transpose(1, 2)

        x3_decoded_output = self.decoder_linear(x3_decoded.transpose(1, 2))
        x3_decoded_output = x3_decoded_output.contiguous().transpose(1, 2)
        
        return (x1_decoded_output, x2_decoded_output, x3_decoded_output), (x1_commit_loss, x2_commit_loss, x3_commit_loss, pitch_commit_loss)
