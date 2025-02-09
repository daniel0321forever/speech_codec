import os, random, sys, logging, time, math, datetime
from abc import ABC, abstractclassmethod

import numpy as np
import pickle
import librosa
from tqdm import tqdm
from pesq import pesq
from scipy.io import wavfile
from parallel_wavegan.utils import load_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchaudio.transforms import Spectrogram


from dataset import NSCDataset
from utils.utils import compute_mel, read_hdf5

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


class Pipeline(ABC):

    def __init__(
            self,
            model: torch.nn.Module,
            weight_dir: str,
            log_dir: str | None,
            device: str | None = None, 
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

        self.optimizer = torch.optim.AdamW(
            [{'params': self.model.parameters(), 'weight_decay': 1e-9},],
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-9
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=0.92,
            verbose=True,
        )

        self.criteria = nn.MSELoss()

        self.loss_len = 2
        self.val_loss_len = 1

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
            batch_size = 16,
            num_workers = 4,
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
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

        self.test_loader = DataLoader(self.test_set,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

        self.writer = SummaryWriter(self.log_dir)

        print("Start training,", time.time())

        for epoch in range(max_epochs):
            
            print("\nepoch |", epoch)

            # training
            self.train_loss = np.zeros(self.loss_len)
            self.model.train()

            for i, batch in tqdm(enumerate(self.train_loader)):    
                self.train_step(i, batch)
            
            # validation
            self.model.eval()
            with torch.no_grad():
                self.val_loss = np.zeros(self.val_loss_len)

                for i, batch in tqdm(enumerate(self.test_loader)):
                    self.val_step(i, batch)
            
            # scheduler
            self.scheduler.step()


            # output step result
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
                self.writer.add_scalars('Loss/mel loss 3 VQ (400bits/s)', {'train': self.train_loss[0],
                                                    'valid': self.val_loss[0]}, epoch + 1)

                self.writer.add_scalar('Loss/VQ#3 training commit loss', self.train_loss[1], epoch + 1)
  

                for i in range(self.loss_len, len(self.train_loss)):
                    self.writer.add_scalar(f'training loss {i}', self.train_loss[i], epoch + 1)      

    def test(
            self,
            test_dir = "/media/daniel0321/LargeFiles/datasets/VCTK/raw_data/test/wav48_silence_trimmed",
            model_name = "100.pth.tar",
    ):
        
        self.test_dir = test_dir
        self.test_scores = np.zeros(3)
        

        # restore model
        self.model_path = os.path.join(self.weight_dir, os.path.basename(model_name))
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()

        # vocoder
        self.vocoder = PWGVocoder(
            device=self.device,
            normalize_path='../../pwg/stats.h5',
            vocoder_path='../../pwg/checkpoint-400000steps.pkl'
        ).to(self.device)
        
        if self.is_speech_resynth:
            
            count = 0
            
            singer_dirs = os.listdir(self.test_dir)
            for singer in singer_dirs:
                singer = os.path.join(self.test_dir, singer)
                source_audios = os.listdir(singer)

                for source_audio in source_audios:
                    source_audio = os.path.join(singer, source_audio)

                    print(source_audio)

                    # get input data
                    self.source_y = librosa.core.load(source_audio, sr=self.sr, mono=True)[0] # might be helpful in inference
                    self.source_y = librosa.util.normalize(self.source_y) * 0.9
                    self.source_mel_feature, self.source_spc = compute_mel(self.source_y)

                    output_wavs = self.inference(source_path=source_audio) # len = 3 normally

                    # get wave file 
                    wavfile.write(f"vq.wav", self.sr, output_wavs)
                    
                    # get vocoder origin wave file
                    vocoder_baseline_input = torch.tensor(self.source_mel_feature).unsqueeze(0).to(self.device)
                    vocoder_baseline_output, _ = self.vocoder(vocoder_baseline_input, output_all=True)
                    vocoder_baseline_output = vocoder_baseline_output[0].cpu().numpy()
                    
                    wavfile.write(f"norm.wav", self.sr, self.source_y)
                    wavfile.write('vocoder.wav', self.sr, vocoder_baseline_output)


                    # load output wav file
                    vocoder_result = [None] * len(self.test_scores)
                    vocoder_result[0], _ = librosa.core.load('vq.wav', sr=16000, mono=True) # vq_result1
                    vocoder_result[1], _ = librosa.core.load('vocoder.wav', sr=16000, mono=True) # vocoder_baseline_result
                    vocoder_result[2], _ = librosa.core.load('norm.wav', sr=16000, mono=True) # orig_audio_16k_norm

                    for i in range(len(vocoder_result)):
                        if len(vocoder_result[i]) > len(vocoder_result[2]):
                            vocoder_result[i] = vocoder_result[i][:len(vocoder_result[2])]
                        
                        self.test_scores[i] += pesq(16000, vocoder_result[2], vocoder_result[i], 'nb')
                        print(pesq(16000, vocoder_result[2], vocoder_result[i], 'nb'), end=' ')

                    for name in ["norm.wav", "vq.wav", "vocoder.wav"]:
                        os.remove(name)
                    
                    count += 1
        
            print("\n -- AVG Scores --")
            for i in range(len(self.test_scores)):
                print(self.test_scores[i] / count)
        
        else:
            logging.error("The testing method should be overwritten when is_speech_resynth == False")
            raise AssertionError
    
    
    def generate(self, audio_path, save_dir, model_name="100.pth.tar"):

        # restore model
        self.model_path = os.path.join(self.weight_dir, os.path.basename(model_name))
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()

        # vocoder
        self.vocoder = PWGVocoder(
            device=self.device,
            normalize_path='../../pwg/stats.h5',
            vocoder_path='../../pwg/checkpoint-400000steps.pkl'
        ).to(self.device)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        # get wav file
        self.source_y = librosa.core.load(audio_path, sr=self.sr, mono=True)[0] # might be helpful in inference
        self.source_y = librosa.util.normalize(self.source_y) * 0.9
        self.source_mel_feature, self.source_spc = compute_mel(self.source_y)

        output_wavs = self.inference(source_path=audio_path) # len = 3 normally

        wavfile.write(os.path.join(save_dir, f"norm.wav"), self.sr, self.source_y)
        wavfile.write(os.path.join(save_dir, f"vq.wav"), self.sr, output_wavs)

    
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
