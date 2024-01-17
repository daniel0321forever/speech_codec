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


class EncoderBlock(nn.Module):

    def __init__(self, heads, latent_dim=128, temp_downsample_rate=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.latent_dim = latent_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=heads, dim_feedforward=latent_dim, norm_first=True)
        self.res_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4, norm=nn.LayerNorm)

        # downsample by using stride = 2 convolution once
        if self.temp_downsample_rate == 2:
            self.downsample_layers = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=2, padding=self.padding, dilation=1),
            )

        # downsample by using stride = 2 convolution twice
        elif self.temp_downsample_rate == 4:
            self.downsample_layers = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=2, padding=self.padding, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, self.kernel_size, stride=2, padding=self.padding, dilation=1),
            )

    def forward(self, x, src_mask):
        
        """
        @parameters
            x: shape(batch, seq, latent_dim) input tensor.
            src_mask: It is provided to make sure the padding element in the input would not be taken into consideration. It would 
            mark the element that is equal to zero (padding element) and mask it to -inf in the Attention block
        """

        # src_mask get its name for being the mask for encoder input
        output = self.res_encoder(x, src_key_padding_mask=src_mask)
        output = output.transpose(1, 2).contiguous()

        # repeat the downsampled elements so that the length is the same
        if self.temp_downsample_rate == 2:
            downsampled = self.downsample_layers(output)
            downsampled = torch.repeat_interleave(downsampled, 2, dim=2)
            output = output.transpose(1, 2).contiguous()
            return downsampled, output

        elif self.temp_downsample_rate == 4:
            downsampled = self.downsample_layers(output)
            downsampled = torch.repeat_interleave(downsampled, 4, dim=2)
            output = output.transpose(1, 2).contiguous()
            return downsampled, output

        else:
            output = output.transpose(1, 2).contiguous()
            return output 

class DecoderBlock(nn.Module):
    def __init__(self, heads, latent_dim):
        super(DecoderBlock, self).__init__()
        
        decoderLayer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=heads, dim_feedforward=latent_dim, norm_first=True)
        self.res_decoder = nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=4, norm=nn.LayerNorm)

    def forward(self, x, enc_output, tgt_mask, tgt_padding_mask=None, enc_output_mask=None):
        """
        @parameters
            tgt_mask: The tgt mask would also handle the padding element in decoder input, while also shield the input that could not be foresee by decoder
            enc_output_mask: The enc_output_mask is for the map for memory
        """
        # propogation
        output = self.res_decoder(tgt=x, memory=enc_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_mask=enc_output_mask)

        return output

class CodecT(nn.Module):
    def __init__(self, device='cpu'):
        super(CodecT, self).__init__()
        
        self.device = device
        self.heads = 4
        self.input_dim = 80 # Yeah, it is the frequencies dimension
        self.latent_dim = 256
        self.combined_dim = self.latent_dim * 3
        self.frames = 80
        self.pitch_dim = 64
        self.mag_dim = 64

        self.encoder_linear = nn.Sequential(
                nn.Conv1d(self.input_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
            )

        self.pitch_encoder_linear = nn.Sequential(
                nn.Conv1d(self.pitch_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
            )

        self.mag_encoder_linear = nn.Sequential(
                nn.Conv1d(self.pitch_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
            )
        
        # We use the combined input here. The input shape is expected to be shape(batch, frames, combined_latent_dim)
        self.res_encoder_block1 = EncoderBlock(heads=4, latent_dim=self.combined_dim, temp_downsample_rate=4)
        self.res_encoder_block2 = EncoderBlock(heads=4, latent_dim=self.combined_dim, temp_downsample_rate=2)
        self.res_encoder_block3 = EncoderBlock(heads=4, latent_dim=self.combined_dim)


        self.vq1 = VectorQuantize(
                        dim = self.combined_dim,
                        codebook_size = 256,     
                        decay = 0.99,      
                        commitment_weight = 1.
                    )

        self.vq2 = VectorQuantize(
                        dim = self.combined_dim,
                        codebook_size = 256,     
                        decay = 0.99,      
                        commitment_weight = 1.
                    )

        self.vq3 = VectorQuantize(
                        dim = self.combined_dim,
                        codebook_size = 256,     
                        decay = 0.99,      
                        commitment_weight = 1.
                    )
        
        # input (frames, combined_dim) -> output (frames, combined_dim)
        self.res_decoder_block1 = DecoderBlock(heads=4, latent_dim=self.combined_dim)
        self.res_decoder_block2 = DecoderBlock(heads=4, latent_dim=self.combined_dim)
        self.res_decoder_block3 = DecoderBlock(heads=4, latent_dim=self.combined_dim)

        # project the vector to the original dim
        self.decoder_linear = nn.Sequential(
                nn.Conv1d(self.combined_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.input_dim, 1, stride=1, padding=0, dilation=1)
            )
    
    
    def generate_mask(self, src, tgt):
        src_pad_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        tgt_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        return src_pad_mask, tgt_pad_mask, tgt_mask
    
    def encoder(self, x, pitch, mag):
        # transmit the compressed values (code book index of each input vector) to decoder
        # transmit the quantized tensor (decoded tensor) to the next level

        x = x.contiguous().transpose(1, 2) # (freq, frames)
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

        # decode pitch, mag
        pitch = self.pitch_encoder(pitch.transpose(1, 2)) # (64, 80)
        pitch_quantized, pitch_indices, pitch_commit_loss = self.vq_pitch(pitch)
        pitch_quantized = pitch_quantized.transpose(1, 2) # (80, 64)
        # mag = self.mag_decoder(mag.transpose(1, 2)) # (64, 80)

        return x1_indices, x2_indices, x3_indices, pitch_indices

    def forward(self, x: torch.Tensor, pitch: torch.Tensor, mag: torch.Tensor):


        # TODO: x should be index encoded
        # TODO: should probably right shift the decoder

        x = self.encoder_linear(x.transpose(1, 2)) # transpose the input into (bins, frames) and perform linear encoding
        x = x.contiguous().transpose(1, 2) # transpose back into (frames, bins)

        pitch = self.pitch_encoder_linear(pitch.transpose(1, 2))
        pitch = pitch.contiguous().transpose(1, 2)

        mag = self.mag_encoder_linear(mag.transpose(1, 2))
        mag = mag.contiguous().transpose(1, 2)

        combined = torch.concat([x, pitch, mag], dim=-1)

        # First CNN + VQ compressor
        src_pad_mask, tgt_pad_mask, tgt_mask = self.generate_mask(x, x)
        ds_x1, x1 = self.res_encoder_block1(x, src_pad_mask) # shape (frames, feats)
        x1_quantized, x1_indices, x1_commit_loss = self.vq1(ds_x1) # (frames, feats)
        x1_quantize_residual = x1 - x1_quantized # shape (frames=80, feats=256)

        src_pad_mask, tgt_pad_mask, tgt_mask = self.generate_mask(x1_quantize_residual, x1_quantize_residual)
        ds_x2, x2 = self.res_encoder_block2(x1_quantize_residual, src_pad_mask)
        x2_quantized, x2_indices, x2_commit_loss = self.vq2(ds_x2)
        x2_quantize_residual = x2 - x2_quantized

        src_pad_mask, tgt_pad_mask, tgt_mask = self.generate_mask(x2_quantize_residual, x2_quantize_residual)
        x3 = self.res_encoder_block3(x2_quantize_residual, src_pad_mask)
        x3_quantized, x3_indices, x3_commit_loss = self.vq3() # shape (frames=80, feats=256)

        
        # decode x1
        src_pad_mask, tgt_pad_mask, tgt_mask = self.generate_mask(x1_quantized, x)
        x1_decoded = self.res_decoder_block3(x, x1_quantized, tgt_mask, tgt_pad_mask, src_pad_mask) # (frames, feats)

        # decode x2:
        src_pad_mask, tgt_pad_mask, tgt_mask = self.generate_mask(x2_quantized, x1_quantize_residual)
        x2_decoded = self.res_decoder_block2(x1_quantize_residual, x2_quantized, tgt_mask, tgt_pad_mask, src_pad_mask) # (frames, feats)
        src_pad_mask, tgt_pad_mask, tgt_mask = self.generate_mask(x2_decoded + x1_decoded, x)
        x2_decoded = self.res_decoder_block2(x, x2_decoded + x1_decoded, tgt_mask, tgt_pad_mask, src_pad_mask) # (frames, feats)

        # decode x3 (The flow written in the report)
        src_pad_mask, tgt_pad_mask, tgt_mask = self.generate_mask(x3_quantized, x2_quantize_residual)
        x3_decoded = self.res_decoder_block2(x2_quantize_residual, x3_quantized, tgt_mask, tgt_pad_mask, src_pad_mask) # (frames, feats)
        src_pad_mask, tgt_pad_mask, tgt_mask = self.generate_mask(x3_decoded + x2_decoded, x1_quantize_residual)
        x3_decoded = self.res_decoder_block2(x1_quantize_residual, x3_decoded + x2_decoded, tgt_mask, tgt_pad_mask, src_pad_mask) # (frames, feats)
        src_pad_mask, tgt_pad_mask, tgt_mask = self.generate_mask(x3_decoded + x1_decoded, x)
        x3_decoded = self.res_decoder_block2(x, x3_decoded + x1_decoded, tgt_mask, tgt_pad_mask, src_pad_mask) # (frames, feats)

        x1_decoded_output = self.decoder_linear(x1_decoded.transpose(1, 2)) # (latent_feats=256, frames)
        x1_decoded_output = x1_decoded_output.contiguous().transpose(1, 2) # (frames, latent_feats)

        x2_decoded_output = self.decoder_linear(x2_decoded.transpose(1, 2))
        x2_decoded_output = x2_decoded_output.contiguous().transpose(1, 2)

        x3_decoded_output = self.decoder_linear(x3_decoded.transpose(1, 2))
        x3_decoded_output = x3_decoded_output.contiguous().transpose(1, 2)
        
        return (x1_decoded_output, x2_decoded_output, x3_decoded_output), (x1_commit_loss, x2_commit_loss, x3_commit_loss, pitch_commit_loss, mag_commit_loss)
    