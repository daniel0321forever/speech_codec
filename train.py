import os, re, random
import numpy as np
import sys, pickle

import time
import math

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Codec, PWGVocoder, MelResCodec
from dataset import NSCDataset

from torchaudio.transforms import Spectrogram
# Since we need to normalize the loss
spec_loss_function = nn.MSELoss()

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)

if __name__ == "__main__":

    gpu_id = '0'
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


    device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')
    
    print(f"Current device: {device}")
    
    
    name = "pitch_and_mag"
    model_dir = f'models/{name}'
    logger_path = f'logger/{name}'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)

    model = Codec(device=device).to(device)
    model.apply(init_weights)
    print("Model Has Been Defined")
    num_param = get_param_num(model)
    print('Number of codec parameters:', num_param)

    learning_rate = 1e-4

    optimizer = torch.optim.Adam([
                                    {'params': model.parameters(), 'weight_decay': 1e-9},
                                ],
                                  lr=learning_rate,
                                  betas=(0.9, 0.999),
                                  eps=1e-9)

    with open('dataset/train_spc.pkl', 'rb') as f:
        train_dataset = pickle.load(f)

    with open('dataset/test_spc.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    print ("Number of training sample:", len(train_dataset))
    print ("Number of test sample:", len(test_dataset))

    # Get Training Loader
    training_loader = DataLoader(train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=4)

    test_loader = DataLoader(test_dataset,
                              batch_size=16,
                              shuffle=False,
                              num_workers=4)

    writer = SummaryWriter(logger_path)

    # vocoder.eval()

    print ("Start training,", time.time())
    for epoch in range(100):
        split_loss = np.zeros(6)
        use_linear_batch = 0
        for i, batchs in tqdm(enumerate(training_loader)):

            model_input = batchs[0].to(device) # take only the mel-cepstrum
            pitch = batchs[2].to(device)
            mag = batchs[3].to(device)


            model_output_list, commit_loss_list = model(model_input, pitch, mag)

            # the output from 3 different bitrate
            mel_loss1 = spec_loss_function(model_output_list[0], model_input)
            mel_loss2 = spec_loss_function(model_output_list[1], model_input)
            mel_loss3 = spec_loss_function(model_output_list[2], model_input)

            t_l = mel_loss1 + mel_loss2 * 0.2 + mel_loss3 * 0.04 + commit_loss_list[0] + commit_loss_list[1] * 0.2 + commit_loss_list[2] * 0.04
            
            # get each loss respectively
            split_loss[0] += mel_loss1.item()
            split_loss[1] += mel_loss2.item()
            split_loss[2] += mel_loss3.item()

            split_loss[3] += commit_loss_list[0].item()
            split_loss[4] += commit_loss_list[1].item()
            split_loss[5] += commit_loss_list[2].item()

            if i % 4000 == 0:
                print('mel loss: {:.4f} {:.4f} {:.4f}; commit loss: {:.4f} {:.4f} {:.4f}'.format(
                        mel_loss1.item(), mel_loss2.item(), mel_loss3.item(),
                        commit_loss_list[0].item(), commit_loss_list[1].item(), commit_loss_list[2].item()))

            t_l.backward()

            # gradient = gradient * coeff (< 1)
            nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)

            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            split_test_loss = np.zeros(3)
            use_linear_batch_test = 0
            for i, batchs in tqdm(enumerate(test_loader)):
                # print (batchs.shape)
                model_input = batchs[0].to(device)
                pitch = batchs[2].to(device)
                mag = batchs[3].to(device)

                model_output_list, commit_loss_list = model(model_input, pitch, mag)

                mel_loss1 = spec_loss_function(model_output_list[0], model_input)
                mel_loss2 = spec_loss_function(model_output_list[1], model_input)
                mel_loss3 = spec_loss_function(model_output_list[2], model_input)

                split_test_loss[0] += mel_loss1.item()
                split_test_loss[1] += mel_loss2.item()
                split_test_loss[2] += mel_loss3.item()
        
        split_loss = split_loss / len(training_loader)
        print ("Training loss:", split_loss)

        split_test_loss = split_test_loss / len(test_loader)
        print ("Test loss:", split_test_loss)

        model.train()
        save_dict = model.state_dict()

        target_model_path = os.path.join(model_dir, str(epoch + 1) + '.pth.tar')
        torch.save(save_dict, target_model_path)

        writer.add_scalars('Loss/mel loss 1 VQ (400bits/s)', {'train': split_loss[0],
                                            'valid': split_test_loss[0]}, epoch + 1)

        writer.add_scalars('Loss/mel loss 2 VQ (1200bits/s)', {'train': split_loss[1],
                                    'valid': split_test_loss[1]}, epoch + 1)

        writer.add_scalars('Loss/mel loss 3 VQ (2800bits/s)', {'train': split_loss[2],
                                    'valid': split_test_loss[2]}, epoch + 1)

        writer.add_scalar('Loss/VQ#1 training commit loss', split_loss[3], epoch + 1)
        writer.add_scalar('Loss/VQ#2 training commit loss', split_loss[4], epoch + 1)
        writer.add_scalar('Loss/VQ#3 training commit loss', split_loss[5], epoch + 1)