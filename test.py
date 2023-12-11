import os, sys, re, time, argparse, logging

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
from dataset import NSCDataset
from pipeline import get_param_num
from utils.utils import positional_encoding, compute_mel, compute_pitch

frame_per_second = 200.0
sample_rate = 24000

def inference(model, source_mel, source_pitch, source_mag, device, use_griffim_lim=False, vocoder=None):

    sample_size = 200
    source_mel_segments = []
    source_pitch_segments = []
    source_mag_segments = []

    for i in range(0, len(source_mel), sample_size):
        start = i
        end = min(i+sample_size, len(source_mel))
        
        if start + 1 < end:
            cur_data = np.array(source_mel[start:end])
            cur_pitch = np.array(source_pitch[start:end])
            cur_mag = np.array(source_mag[start:end])

            if end - start < sample_size:
                padding_length = sample_size - (end - start)
                # print (cur_data.shape)
                # cur_data = np.array(np.pad(cur_data, pad_width=((0, padding_length), (0, 0)), constant_values=-10.0))
                cur_data = np.array(np.pad(cur_data, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                cur_pitch = np.array(np.pad(cur_pitch, pad_width=((0, padding_length), (0, 0)), constant_values=0))
                cur_mag = np.array(np.pad(cur_mag, pad_width=((0, padding_length), (0, 0)), constant_values=0))

            print("cur shape")
            print(cur_data.shape, cur_pitch.shape, cur_mag.shape)

            source_mel_segments.append(cur_data)
            source_pitch_segments.append(cur_pitch)
            source_mag_segments.append(cur_mag)
            
    pred_list_vq1 = []
    pred_list_vq2 = []
    pred_list_vq3 = []

    with torch.no_grad():
        for i in range(len(source_mel_segments)):
            model_input = torch.tensor(source_mel_segments[i]).unsqueeze(0).to(device)
            pitch_input = positional_encoding(torch.tensor(source_pitch_segments[i]).unsqueeze(0)).to(device)
            mag_input = torch.tensor(source_mag_segments[i]).unsqueeze(0).to(device)
            
            model_input = torch.log(model_input + 1e-10)
            
            model_output_list, commit_loss_list = model(model_input, pitch_input, mag_input)

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

    return output_wav1, output_wav2, output_wav3

from pesq import pesq

if __name__ == "__main__":
    audio_dir = "/media/daniel0321/LargeFiles/datasets/VCTK/raw_data/test/wav48_silence_trimmed"
    dirs = os.listdir(audio_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=False, default="models/pitch_and_mag/100.pth.tar")
    parser.add_argument('-p', '--prefix', required=False, default="new")
    parser.add_argument('-g', '--gpu_id', required=False, default="0")

    args = parser.parse_args()
    model_path = args.model
    output_prefix = args.prefix
    gpu_id = args.gpu_id

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


    count = 0
    scores1 = []
    scores2 = []
    scores3 = []
    scores_voc = []
    scores_control = []

    for singer in dirs:
        singer = os.path.join(audio_dir, singer)
        
        print(singer)
        source_audios = os.listdir(singer)

        for source_audio in source_audios:
            source_audio = os.path.join(singer, source_audio)

            # Source audio
            source_y, sr = librosa.core.load(source_audio, sr=24000, mono=True)
            source_y = librosa.util.normalize(source_y) * 0.9
            source_mel_feature, source_spc = compute_mel(source_y)

            # source pitch and mag
            source_pitch, source_mag = compute_pitch(source_y)
            max_mag_idx = np.argmax(source_mag, axis=1)    
            source_pitch = np.array([[source_pitch[frame][max_mag_idx[frame]]] for frame in range(len(source_pitch))])
            source_mag = np.array([[source_mag[frame][max_mag_idx[frame]]] for frame in range(len(source_mag))])

            # load Codec model and vocoder
            model = Codec(device=device).to(device)
            num_param = get_param_num(model)
            print('Number of codec parameters:', num_param)

            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print("---Model Restored---", model_path, "\n")
            
            vocoder = PWGVocoder(device=device, normalize_path='pwg/stats.h5'
                , vocoder_path='pwg/checkpoint-400000steps.pkl').to(device)

            model.eval()
            
            # get inference
            use_griffim_lim = False
            output_wav1, output_wav2, output_wav3 = inference(model, source_mel_feature, source_pitch, source_mag, device, use_griffim_lim=use_griffim_lim, vocoder=vocoder)

            # write inference to wav file
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


            # load output wav file
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
            
            # evaluate pesq score 
            scores1.append(pesq(16000, orig_audio_16k_norm, vq_result1, 'nb'))
            scores2.append(pesq(16000, orig_audio_16k_norm, vq_result2, 'nb'))
            scores3.append(pesq(16000, orig_audio_16k_norm, vq_result3, 'nb'))
            scores_voc.append(pesq(16000, orig_audio_16k_norm, vocoder_baseline_result, 'nb'))
            scores_control.append(pesq(16000, orig_audio_16k_norm, orig_audio_16k_norm, 'nb'))

            print(f"\n{source_audio}")
            print(f"PESQ1 (No residual):{scores1[-1]}")
            print(f"PESQ2 (add layer 2 residual data):{scores2[-1]}")
            print(f"PESQ3 (add layer 2, 3 residual data):{scores3[-1]}")
            print(f"No resynthesis vocoder:{scores_voc[-1]}")
            print(f"Max PESQ score:{scores_control[-1]}")

            if count != 18:
                for tail in ["_norm.wav", "_vq1.wav", "_vq2.wav", "_vq3.wav", "_vocoder.wav"]:
                    os.remove(output_prefix + tail)
            
            count += 1

    sample_len = len(scores1)
    print("\n==================AVG Score==================")
    print (f"PESQ1 (No residual):{np.array(scores1).mean()}, PESQ2 (add layer 2 residual data) {np.array(scores2).mean()}, PESQ3 (add layer 2, 3 residual data{np.array(scores3).mean()}\n No resynthesis vocoder{np.array(scores_voc).mean()}, Max PESQ score: {np.array(scores_control).mean()}")
