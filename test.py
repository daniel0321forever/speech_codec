import os, sys, re, time

import pickle
import librosa

import numpy as np
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from model import Codec, PWGVocoder
from dataset import NSCDataset, compute_mel
from train import get_param_num

frame_per_second = 200.0
sample_rate = 24000

def inference(model, source_mel, device, use_griffim_lim=False, vocoder=None):

    sample_size = 200
    source_mel_segments = []

    print("source_mel", source_mel.shape)
    for i in range(0, len(source_mel), sample_size):
        start = i
        end = min(i+sample_size, len(source_mel))
        
        if start + 1 < end:
            cur_data = np.array(source_mel[start:end])
            if end - start < sample_size:
                padding_length = sample_size - (end - start)
                # print (cur_data.shape)
                # cur_data = np.array(np.pad(cur_data, pad_width=((0, padding_length), (0, 0)), constant_values=-10.0))
                cur_data = np.array(np.pad(cur_data, pad_width=((0, padding_length), (0, 0)), constant_values=0))
            
            print(cur_data.shape)
            source_mel_segments.append(cur_data)
            
    pred_list_vq1 = []
    pred_list_vq2 = []
    pred_list_vq3 = []

    with torch.no_grad():
        for i in tqdm(range(len(source_mel_segments))):
            model_input = torch.tensor(source_mel_segments[i]).unsqueeze(0).to(device)
            model_input = torch.log(model_input + 1e-10)
            model_output_list, commit_loss_list = model(model_input)

            if use_griffim_lim:
                waveform_output_vq1 = librosa.griffinlim(torch.exp(model_output_list[0]).squeeze(0).cpu().numpy().T, n_iter=32, hop_length=300)
                waveform_output_vq2 = librosa.griffinlim(torch.exp(model_output_list[1]).squeeze(0).cpu().numpy().T, n_iter=32, hop_length=300)
                waveform_output_vq3 = librosa.griffinlim(torch.exp(model_output_list[2]).squeeze(0).cpu().numpy().T, n_iter=32, hop_length=300)
                pred_list_vq1.append(waveform_output_vq1)
                pred_list_vq2.append(waveform_output_vq2)
                pred_list_vq3.append(waveform_output_vq3)
            else:
                waveform_output_vq1, _ = vocoder(model_output_list[0], output_all=True)
                waveform_output_vq2, _ = vocoder(model_output_list[1], output_all=True)
                waveform_output_vq3, _ = vocoder(model_output_list[2], output_all=True)
                pred_list_vq1.append(waveform_output_vq1[0].cpu())
                pred_list_vq2.append(waveform_output_vq2[0].cpu())
                pred_list_vq3.append(waveform_output_vq3[0].cpu())

    output_wav1 = np.concatenate(pred_list_vq1, axis=0)
    output_wav2 = np.concatenate(pred_list_vq2, axis=0)
    output_wav3 = np.concatenate(pred_list_vq3, axis=0)

    print (output_wav1.shape, output_wav2.shape, output_wav3.shape)

    return output_wav1, output_wav2, output_wav3

from pesq import pesq

if __name__ == "__main__":
    model_path = sys.argv[1]
    source_audio = sys.argv[2]
    output_prefix = sys.argv[3]
    gpu_id = sys.argv[4]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Source audio
    source_y, sr = librosa.core.load(source_audio, sr=24000, mono=True)
    source_y = librosa.util.normalize(source_y) * 0.9
    source_mel_feature, source_spc = compute_mel(source_y)
    
    print (source_y.shape, source_mel_feature.shape)

    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    model = Codec(device=device).to(device)
    num_param = get_param_num(model)
    print('Number of codec parameters:', num_param)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    print("---Model Restored---", model_path, "\n")
    
    # vocoder = PWGVocoder(device=device, normalize_path='train_nodev_all_vctk_parallel_wavegan.v1/stats.h5'
    #     , vocoder_path='train_nodev_all_vctk_parallel_wavegan.v1/checkpoint-400000steps.pkl').to(device)

    model.eval()

    use_griffim_lim = True
    output_wav1, output_wav2, output_wav3 = inference(model, source_mel_feature, device, use_griffim_lim=use_griffim_lim)
    print("get inference")
    wavfile.write(output_prefix + '_vq1.wav', 24000, output_wav1)
    wavfile.write(output_prefix + '_vq2.wav', 24000, output_wav2)
    wavfile.write(output_prefix + '_vq3.wav', 24000, output_wav3)
    wavfile.write(output_prefix + '_norm.wav', 24000, source_y)

    if use_griffim_lim:
        vocoder_baseline_input = torch.tensor(source_mel_feature).unsqueeze(0).to(device)
        vocoder_baseline_output = librosa.griffinlim(torch.exp(vocoder_baseline_input).squeeze(0).cpu().numpy().T, n_iter=32, hop_length=300)
        vocoder_baseline_output = vocoder_baseline_output

    else:
        vocoder_baseline_input = torch.tensor(source_mel_feature).unsqueeze(0).to(device)
        vocoder_baseline_output, _ = vocoder(vocoder_baseline_input, output_all=True)
        vocoder_baseline_output = vocoder_baseline_output[0].cpu().numpy()

    # vocoder_baseline_output = librosa.griffinlim(source_spc.T, n_iter=32, hop_length=300)
    wavfile.write(output_prefix + '_vocoder.wav', 24000, vocoder_baseline_output)


    orig_audio_16k_norm, _ = librosa.core.load(output_prefix + '_norm.wav', sr=16000, mono=True)
    vq_result1, _ = librosa.core.load(output_prefix + '_vq1.wav', sr=16000, mono=True)
    vq_result2, _ = librosa.core.load(output_prefix + '_vq2.wav', sr=16000, mono=True)
    vq_result3, _ = librosa.core.load(output_prefix + '_vq3.wav', sr=16000, mono=True)

    vocoder_baseline_result, _ = librosa.core.load(output_prefix + '_vocoder.wav', sr=16000, mono=True)

    if len(vq_result1) > len(orig_audio_16k_norm):
        vq_result1 = vq_result1[:len(orig_audio_16k_norm)]

    if len(vq_result2) > len(orig_audio_16k_norm):
        vq_result2 = vq_result2[:len(orig_audio_16k_norm)]

    if len(vq_result3) > len(orig_audio_16k_norm):
        vq_result3 = vq_result3[:len(orig_audio_16k_norm)]

    if len(vocoder_baseline_result) > len(orig_audio_16k_norm):
        vocoder_baseline_result = vocoder_baseline_result[:len(orig_audio_16k_norm)]

    print (orig_audio_16k_norm.shape, vq_result1.shape, vq_result2.shape, vq_result3.shape, vocoder_baseline_result.shape)
    
    pesq1 = pesq(16000, orig_audio_16k_norm, vq_result1, 'nb')
    pesq2 = pesq(16000, orig_audio_16k_norm, vq_result2, 'nb')
    pesq3 = pesq(16000, orig_audio_16k_norm, vq_result3, 'nb')
    pesq_vocoder = pesq(16000, orig_audio_16k_norm, vocoder_baseline_result, 'nb')

    pesq_control = pesq(16000, orig_audio_16k_norm, orig_audio_16k_norm, 'nb')

    print (pesq1, pesq2, pesq3, pesq_vocoder, pesq_control)