import os, random, sys, logging, time, math
from abc import ABC, abstractclassmethod

import numpy as np
import pickle
import librosa
from tqdm import tqdm
from pesq import pesq
from scipy.io import wavfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchaudio.transforms import Spectrogram


from model import PWGVocoder
from dataset import NSCDataset
from utils.utils import compute_mel


class Pipeline(ABC):

    def __init__(
            self,
            model: torch.nn.Module,
            weight_dir: str,
            log_dir: str | None,
            device: str | None, 
            gpu_id = "0",
            is_speech_resynth = True,
            sr = 24000,
    ):
        """
        #### Params:
            - weight_dir: The path for trained weight. In training method it would be the output path for model, in testing method it would be the path where we obtain the model's weight.
            - log_dir: The directory that stores the information of training process in tensorboard format
            - model: The torch model module that define the structore of the model
            - device: 'cuda' or 'cpu', define the device to execute the computation
            - gpu_id: The GPU ID used to execute the computation. Default 0 (if there is only one GPU)
        """

        torch.multiprocessing.set_sharing_strategy('file_system')
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        else:
            self.device = device

        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        self.sr = sr
        self.is_speech_resynth = is_speech_resynth
        self.weight_dir = weight_dir
        self.log_dir = log_dir
        self.model = model.to(self.device)
        self.model.apply(self._init_weights)

        print("Model built")
        print(f'| device: {self.device}')
        print("| num of params", self._get_param_num())
        print(f"| weight directory: {self.weight_dir}")
        print(f"| log directory: {self.log_dir}")

    def train(
            self,
            train_set = None,
            test_set = None,
            train_set_path: str | None = None,
            test_set_path: str | None = None,
            batch_size: int | None = None,
            num_workers: int | None = None,
            learning_rate = 1e-4,
            max_epochs = 100,
            save_best = False,
    ):
        if train_set is not None:
            self.train_set = train_set
        elif train_set_path is not None:
            try:
                with open(train_set_path, 'rb') as f:
                    self.train_set = pickle.load(f)
            except EOFError as e:
                logging.error(e)
                exit()
        else:
            logging.error("both train_set and train_set_path are not given values")
            raise ValueError


        if test_set is not None:
            self.test_set = test_set
        elif test_set_path is not None:
            try:
                with open(test_set_path, 'rb') as f:
                    self.test_set = pickle.load(f)
            except EOFError as e:
                logging.error(e)
                exit()
        else:
            logging.error("both train_set and train_set_path are not given values")
            raise ValueError

        self.train_loader = DataLoader(self.train_set,
                                    batch_size=16,
                                    shuffle=True,
                                    num_workers=4)

        self.test_loader = DataLoader(self.test_set,
                                batch_size=16,
                                shuffle=False,
                                num_workers=4)
        
        self.optimizer = torch.optim.Adam(
            [{'params': self.model.parameters(), 'weight_decay': 1e-9},],
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-9
        )
        self.writer = SummaryWriter(self.log_dir)

        self.criteria = nn.MSELoss()
        
        print("Start training,", time.time())

        for epoch in range(max_epochs):
            
            print("epoch |", epoch)
            self.train_loss = np.zeros(6)
            self.model.train()

            for i, batch in tqdm(enumerate(self.train_loader)):    
                self.train_step(i, batch)
            
            self.model.eval()
            with torch.no_grad():
                self.val_loss = np.zeros(3)

                for i, batch in tqdm(enumerate(self.test_loader)):
                    self.val_step(i, batch)
            
            self.train_loss = self.train_loss / len(self.train_loader)
            self.val_loss = self.val_loss / len(self.test_loader)
            print("Training loss:", self.train_loss)
            print("Val loss:", self.val_loss)

            # model saving
            save_dict = self.model.state_dict()
            torch.save(save_dict, os.path.join(self.weight_dir, str(epoch+1)+".pth.tar"))

            # write log
            if not self.is_speech_resynth:
                self.log()

            else:
                self.writer.add_scalars('Loss/mel loss 1 VQ (400bits/s)', {'train': self.train_loss[0],
                                                    'valid': self.val_loss[0]}, epoch + 1)

                self.writer.add_scalars('Loss/mel loss 2 VQ (1200bits/s)', {'train': self.train_loss[1],
                                            'valid': self.val_loss[1]}, epoch + 1)

                self.writer.add_scalars('Loss/mel loss 3 VQ (2800bits/s)', {'train': self.train_loss[2],
                                            'valid': self.val_loss[2]}, epoch + 1)

                self.writer.add_scalar('Loss/VQ#1 training commit loss', self.train_loss[3], epoch + 1)
                self.writer.add_scalar('Loss/VQ#2 training commit loss', self.train_loss[4], epoch + 1)
                self.writer.add_scalar('Loss/VQ#3 training commit loss', self.train_loss[5], epoch + 1)                

    def test(
            self,
            test_dir = "/media/daniel0321/LargeFiles/datasets/VCTK/raw_data/test/wav48_silence_trimmed",
            model_name = "100.pth.tar",
            prefix = f"model_{time.time()}",
    ):
        
        self.test_dir = test_dir
        self.test_scores = np.zeros(5)
        self.prefix = prefix
        
        if self.is_speech_resynth:
            
            count = 0
            
            singer_dirs = os.listdir(self.test_dir)
            for singer in singer_dirs:
                singer = os.path.join(self.test_dir, singer)
                source_audios = os.listdir(singer)

                for source_audio in source_audios:
                    source_audio = os.path.join(singer, source_audio)

                    print(source_audio)

                    # restore model
                    self.model_path = os.path.join(self.weight_dir, os.path.basename(model_name))
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint, strict=True)
                    self.model.eval()

                    # vocoder
                    self.vocoder = PWGVocoder(
                        device=self.device,
                        normalize_path='pwg/stats.h5',
                        vocoder_path='pwg/checkpoint-400000steps.pkl'
                    ).to(self.device)

                    # get input data
                    self.source_y = librosa.core.load(source_audio, sr=self.sr, mono=True)[0] # might be helpful in inference
                    self.source_y = librosa.util.normalize(self.source_y) * 0.9
                    self.source_mel_feature, self.source_spc = compute_mel(self.source_y)

                    output_wavs = self.inference(source_path=source_audio) # len = 3 normally

                    # get wave file 

                    for i in range(len(output_wavs)):
                        wavfile.write(prefix + f"_vq{i+1}.wav", self.sr, output_wavs[i])
                    
                    # get vocoder origin wave file
                    vocoder_baseline_input = torch.tensor(self.source_mel_feature).unsqueeze(0).to(self.device)
                    vocoder_baseline_output, _ = self.vocoder(vocoder_baseline_input, output_all=True)
                    vocoder_baseline_output = vocoder_baseline_output[0].cpu().numpy()
                    
                    wavfile.write(prefix + f"_norm.wav", self.sr, self.source_y)
                    wavfile.write(prefix + '_vocoder.wav', self.sr, vocoder_baseline_output)


                    # load output wav file
                    vocoder_result = [None] * len(self.test_scores)
                    vocoder_result[0], _ = librosa.core.load(prefix + '_vq1.wav', sr=16000, mono=True) # vq_result1
                    vocoder_result[1], _ = librosa.core.load(prefix + '_vq2.wav', sr=16000, mono=True) # vq_result2
                    vocoder_result[2], _ = librosa.core.load(prefix + '_vq3.wav', sr=16000, mono=True) # vq_result3
                    vocoder_result[3], _ = librosa.core.load(prefix + '_vocoder.wav', sr=16000, mono=True) # vocoder_baseline_result
                    vocoder_result[4], _ = librosa.core.load(prefix + '_norm.wav', sr=16000, mono=True) # orig_audio_16k_norm

                    for i in range(len(vocoder_result)):
                        if len(vocoder_result[i]) > len(vocoder_result[4]):
                            vocoder_result[i] = vocoder_result[i][:len(vocoder_result[4])]
                        
                        self.test_scores[i] += pesq(16000, vocoder_result[4], vocoder_result[i], 'nb')
                        print(pesq(16000, vocoder_result[4], vocoder_result[i], 'nb'), end=' ')

                    for tail in ["_norm.wav", "_vq1.wav", "_vq2.wav", "_vq3.wav", "_vocoder.wav"]:
                        os.remove(prefix + tail)
                    
                    count += 1
        
            print("\n -- AVG Scores --")
            for i in range(len(self.test_scores)):
                print(self.test_scores[i] / count)
        
        else:
            logging.error("The testing method should be overwrited when is_speech_resynth == False")
            raise AssertionError
    
    @abstractclassmethod
    def train_step(self, idx, batch):
        """
        The abstract method would perform the training process in each step given the batch
        data, including loss finding and backward propergation.
        """
        pass

    @abstractclassmethod
    def val_step(self, idx, batch):
        """
        The abstract method would perform the validation process in each validation step given
        the val_batch data, including loss finding. The with torch.no_grad() and model.eval() has
        already been done. It would not hurt you to do it again but it is puerly the waste of time.
        On the otherside, the summary of validation loss would also be presented by default.
        """
        pass
    
    def inference(self, source_path: str):
        """
        In the method, one should define the input of model given the source audio path. Then return the model outcome wave data
        for those inputs. The model could be found as self.model and the vocoder could also be found as self.vocoder. This
        method is designed specifically for speech_resynth case, for other case one should directly overwrite the test method.

        Returns:
            - A tuple list of model output
        
        Param:
            - source_path: The path to raw audio source for testing, one should use it to extract required audio features
        """
        pass

    def log(self, epoch):
        """
        The function is for you to design your own logging format on tensorboard, one could use the 
        self.writer to log the events in training process in this method.
        """
        pass

    def _get_param_num(self):
        num_param = sum(param.numel() for param in self.model.parameters())
        return num_param

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
