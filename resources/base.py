import os, random, sys, logging, time, math
from abc import ABC, abstractclassmethod

import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import Spectrogram


from model import Codec, PWGVocoder, MelResCodec
from dataset import NSCDataset
from resources.utils import positional_encoding


class Model(ABC):

    def __init__(
            self,
            model: torch.nn.Module,
            weight_dir: str,
            log_dir = None | str,
            device = None | str, 
            gpu_id = "0",
            is_speech_resynth = True,
    ):
        """
        #### Params:
        weight_dir: The path for trained weight. In training method it would be the output path for model, in testing method it would be the path where we obtain the model's weight.
        log_dir: The directory that stores the information of training process in tensorboard format
        model: The torch model module that define the structore of the model
        device: 'cuda' or 'cpu', define the device to execute the computation
        gpu_id: The GPU ID used to execute the computation. Default 0 (if there is only one GPU)
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
        
        self.is_speech_resynth = is_speech_resynth
        self.weight_dir = weight_dir
        self.log_dir = log_dir
        self.model = model.to(self.device)
        self.model.apply(self._init_weights)

        logging.info("Model built")
        logging.info(f'| device: {self.device}')
        logging.info("| num of params", self._get_param_num())
        logging.info(f"| weight directory: {self.weight_dir}")
        logging.info(f"| log directory: {self.log_dir}")

    def train(
            self,
            train_set = None,
            test_set = None,
            train_set_path = None | str,
            test_set_path = None | str,
            batch_size = 16 | int,
            num_workers = 1 | int,
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
        
        logging.info("Start training,", time.time())

        for epoch in range(max_epochs):
            
            logging.info("epoch |", epoch)
            self.train_loss = np.zeros(6)
            self.model.train()

            for i, batch in tqdm(enumerate(self.train_loader)):    
                self.train_step()
            
            self.model.eval()
            with torch.no_grad():
                self.val_loss = np.zeros(3)

                for i, batch in tqdm(enumerate(self.test_loader)):
                    self.val_step(batch)
            
            self.train_loss = self.train_loss / len(self.train_loader)
            self.val_loss = self.val_loss / len(self.test_loader)
            logging.info("Training loss:", self.train_loss)
            logging.info("Val loss:", self.val_loss)

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

    def test():
        pass
    
    @abstractclassmethod
    def train_step(self, batch):
        """
        The abstract method would perform the training process in each step given the batch
        data, including loss finding and backward propergation.
        """
        pass

    @abstractclassmethod
    def val_step(self, batch):
        """
        The abstract method would perform the validation process in each validation step given
        the val_batch data, including loss finding. The with torch.no_grad() and model.eval() has
        already been done. It would not hurt you to do it again but it is puerly the waste of time.
        On the otherside, the summary of validation loss would also be presented by default.
        """
        pass
    
    @abstractclassmethod
    def test_step(self):
        """
        The functiion should be overited by subclass
        """
        pass
    
    @abstractclassmethod
    def log(self):
        """
        The function is for you to design your own logging format on tensorboard, one could use the 
        self.writer to log the events in training process in this abstract method.
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