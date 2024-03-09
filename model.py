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

from utils.utils import PositionalEncoding


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

    def __init__(self, heads, latent_dim=128, temp_downsample_rate=1, num_layers=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.latent_dim = latent_dim
        self.kernel_size = 3
        self.padding = (self.kernel_size - 1) // 2
        self.temp_downsample_rate = temp_downsample_rate

        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=heads, dim_feedforward=latent_dim, norm_first=True, batch_first=True)
        layer_norm = nn.LayerNorm(self.latent_dim)
        self.res_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=layer_norm)

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

    def forward(self, x, src_mask=None):
        
        """
        @parameters
            x: shape(batch, seq, latent_dim) input tensor.
            src_mask: It is provided to make sure the padding element in the input would not be taken into consideration. It would 
            mark the element that is equal to zero (padding element) and mask it to -inf in the Attention block
        """

        # src_mask get its name for being the mask for encoder input
        output = self.res_encoder(x, src_mask)
        output = output.transpose(1, 2).contiguous()

        # repeat the downsampled elements so that the length is the same
        if self.temp_downsample_rate == 2:
            downsampled = self.downsample_layers(output)
            downsampled = torch.repeat_interleave(downsampled, 2, dim=2)
            downsampled = downsampled.transpose(1, 2).contiguous()
            output = output.transpose(1, 2).contiguous()
            return downsampled, output

        elif self.temp_downsample_rate == 4:
            downsampled = self.downsample_layers(output)
            downsampled = torch.repeat_interleave(downsampled, 4, dim=2)
            downsampled = downsampled.transpose(1, 2).contiguous()
            output = output.transpose(1, 2).contiguous()
            return downsampled, output

        else:
            output = output.transpose(1, 2).contiguous()
            return output 

class DecoderBlock(nn.Module):
    def __init__(self, heads, latent_dim, num_layers=4):
        super(DecoderBlock, self).__init__()
        self.latent_dim = latent_dim
        
        decoderLayer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=heads, dim_feedforward=latent_dim, norm_first=True, batch_first=True)
        layer_norm = nn.LayerNorm(self.latent_dim)
        self.res_decoder = nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=num_layers, norm=layer_norm)

    def forward(self, x, enc_output, tgt_mask, tgt_pad_mask=None, src_pad_mask=None):
        """
        @parameters
            tgt_mask: The tgt mask would also handle the padding element in decoder input, while also shield the input that could not be foresee by decoder
            src_pad_mask: The src_pad_mask is for the map for memory
        """
        # propogation
        output = self.res_decoder(tgt=x, memory=enc_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_mask=src_pad_mask)

        return output

class CodecT(nn.Module):

    """
    
    In this version, we directly use the mel data "x" for each kind of decoder. The ideal effect is that the decoder would output
    the residual part based on the encoder-output content while the input "x" is all the same.

    By using x directly as decoder input, we can use the output from decoder to get form the decoder input during autoregressive
    testing.
    
    """

    def __init__(self, device='cuda'):
        super(CodecT, self).__init__()

        
        self.device = device
        self.heads = 4
        self.frames = 80
        self.batch_size = 16
        
        self.mel_dim = 80 # input vector <=> output vector
        self.pitch_dim = 1024 # use config['input_dim]
        self.latent_dim = 256 # the encoder output should have the same dim as decoder input = output_dim = input_dim
        
        self.transformer_layer = 6

        self.bos = - torch.ones(self.batch_size, 1, self.latent_dim).to(self.device) # the input for the decoder would be the output from the model at testing stage

        mask = (torch.triu(torch.ones(self.frames, self.frames)) == 1).transpose(0, 1).to(self.device)
        self.mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # encoding
        self.encoder_linear = nn.Sequential(
                nn.Conv1d(self.mel_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
            )
        
        self.res_encoder_block = EncoderBlock(heads=self.heads, latent_dim=self.latent_dim, num_layers=self.transformer_layer)

        # vq
        self.vq = VectorQuantize(
                        dim = self.latent_dim,
                        codebook_size = 1024,     
                        decay = 0.99,      
                        commitment_weight = 1.
                    )
        # decoding
        self.res_decoder_block = DecoderBlock(heads=self.heads, latent_dim=self.latent_dim, num_layers=self.transformer_layer)

        
        # add pitch data
        self.pitch_encoder_linear = nn.Sequential(
                nn.Conv1d(self.pitch_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
            )
        
        # project the vector to the original dim
        self.decoder_linear = nn.Sequential(
                nn.Conv1d(self.latent_dim + self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.latent_dim, 1, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(),
                nn.Conv1d(self.latent_dim, self.mel_dim, 1, stride=1, padding=0, dilation=1)
            )
    
    def right_shift(self, x: torch.Tensor):
        bos = self.bos[:x.shape[0]]
        right_shifted = torch.concat([bos, x[:, :-1]], dim=1)
        return right_shifted

    
    def encoder(self, x):

        encoder_input = self.encoder_linear(x.transpose(1, 2)) # transpose the input into (bins, frames) and perform linear encoding
        encoder_input = encoder_input.contiguous().transpose(1, 2) # transpose back into (frames, bins)


        # res encode
        x_encoded = self.res_encoder_block(encoder_input) # shape (frames, feats)
        x_quantized, x_indices, x_commit_loss = self.vq(x_encoded) # (frames, feats)

        return x_quantized

    def decoder(self, pitch_input, mel_input, x_quantized):
        
        """
        Decode and output the mel vector with the help of quantized pitch information

        @param
        pitch_input: the positional encoded batch of pitch
        mel_input: right shifted time series of mel data, starting of bos
        """

        # mask
        mask = (torch.triu(torch.ones(mel_input.shape[1], mel_input.shape[1])) == 1).transpose(0, 1)
        self.mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(self.device)

        # res decode
        x_decoded = self.res_decoder_block(mel_input, x_quantized, self.mask) # (frames, feats)

        # pitch encode
        pitch = self.pitch_encoder_linear(pitch_input.transpose(1, 2))  # transpose the input into (pitches, frames) and perform linear encoding
        pitch = pitch.contiguous().transpose(1, 2) # transpose back into (frames, pitch)

        # combine vector
        combined = torch.concat([x_decoded, pitch], dim=-1)
        x_decoded_output = self.decoder_linear(combined.transpose(1, 2)) # (latent_feats=256*3, frames)
        x_decoded_output = x_decoded_output.contiguous().transpose(1, 2) # (frames, latent_feats)

        
        return x_decoded_output

    def forward(self, x: torch.Tensor, pitch: torch.Tensor):

        """
        @input
        x: The positional encoded mel time series, shape([batch, frames, bins])
        pitch: The positional encoded pitch index, shape([batch, frames, 1024])

        @output
        The mel vector (x)

        Note: The mel input is right shifted, while the pitch is not right shifted
        """

        encoder_input = self.encoder_linear(x.transpose(1, 2)) # transpose the input into (bins, frames) and perform linear encoding
        encoder_input = encoder_input.contiguous().transpose(1, 2) # transpose back into (frames, bins)


        # res encode
        x_encoded = self.res_encoder_block(encoder_input) # shape (frames, feats)
        x_quantized, x_indices, x_commit_loss = self.vq(x_encoded) # (frames, feats)

        # res decode
        encoded_decoded_input = self.encoder_linear(x.transpose(1, 2))
        encoded_decoded_input = encoded_decoded_input.contiguous().transpose(1, 2)

        x_decoded = self.res_decoder_block(self.right_shift(encoded_decoded_input), x_quantized, self.mask) # (frames, feats)


        # pitch encode
        pitch = self.pitch_encoder_linear(pitch.transpose(1, 2))  # transpose the input into (pitches, frames) and perform linear encoding
        pitch = pitch.contiguous().transpose(1, 2) # transpose back into (frames, pitch)

        # combine vector
        combined = torch.concat([x_decoded, pitch], dim=-1)
        x_decoded_output = self.decoder_linear(combined.transpose(1, 2)) # (latent_feats=256*3, frames)
        x_decoded_output = x_decoded_output.contiguous().transpose(1, 2) # (frames, latent_feats)

        return x_decoded_output, x_commit_loss