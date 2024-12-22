import numpy as np
import torch 
from torch.utils.data import Dataset
import torch.nn as nn
import torchaudio
import librosa



def amplitude_to_decibel(x, amin=1e-10, dynamic_range=80.0):
    """[K] Convert (linear) amplitude to decibel (log10(x)).

    x: Keras *batch* tensor or variable. It has to be batch because of sample-wise `K.max()`.

    amin: minimum amplitude. amplitude smaller than `amin` is set to this.

    dynamic_range: dynamic_range in decibel

    """
    log_spec = 10 * torch.log(torch.maximum(x, torch.ones_like(x)*amin)) / torch.log(torch.tensor([10]))
    log_spec = log_spec - torch.max(log_spec)  # [-?, 0]
    log_spec = torch.maximum(log_spec, torch.ones_like(log_spec)-1 * dynamic_range)  # [-80, 0]
    return log_spec

class SpeechDataset(Dataset):
    def __init__(self,
                  list_IDs,
                  labels,
                  dim=16000,
                  ts_ann=0.008,
                  create_all_mels=True,
                  train_or_val='train',
                  mean=None,
                  std=None,
                  hop_length=128,
                  n_fft=1024):
        # here is a mapping from this index to the mother ds index
        self.dim = dim
        self.labels = labels
        self.list_IDs = list_IDs
        self.train_or_val = train_or_val
        self.mean = mean
        self.std = std
        self.sr = dim
        self.n_fft = n_fft
        self.n_mels = 80
        self.f_min = 40
        self.f_max = self.sr/2
        self.mel_basis = torch.from_numpy(librosa.filters.mel(sr=dim, n_fft=self.n_fft, n_mels=self.n_mels,
                               fmin=self.f_min, fmax=self.f_max,
                               htk=False, norm=1))
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft,hop_length=hop_length,win_length=self.n_fft,power=2,center=True)
        self.create_all_mels = create_all_mels
        if self.create_all_mels:
            self.create_mels()  

    def melspec(self,x):
        new_spec = self.spec(x)
        new_mel = torch.matmul(self.mel_basis,new_spec)
        new_mel = torch.sqrt(new_mel)
        new_mel = amplitude_to_decibel(new_mel)
        #pads 0
        new_mel = new_mel[...,1:]
        return new_mel


    def get_mel(self,input_indices):
        torch_inputs = torch.empty(len(input_indices),self.dim)
        for i,input_idx in enumerate(input_indices):
            ID = self.list_IDs[input_idx]
            curX = torch.from_numpy(np.load(ID)).float()
            X = torch.zeros((self.dim)).float()
            #curX could be bigger or smaller than self.dim
            if curX.shape[0] == self.dim:
                X = curX
                #print('Same dim')
            elif curX.shape[0] > self.dim: #bigger
                #we can choose any position in curX-self.dim
                X = curX[-self.dim:]
                #print('File dim bigger')
            else: #smaller
                X[-curX.shape[0]:] = curX
                #print('File dim smaller')
            torch_inputs[i] = X
        torch_inputs = self.melspec(torch_inputs)
        return torch_inputs

    def normalize(self,x):
        mean = torch.mean(x,[1,2],True)
        std = torch.std(x,[1,2],True,True)
        x = (x - mean)/(std + 1e-10)
        return x

    def create_mels(self):
        print('Creating the mel dataset')
        first_input = self.get_mel([0])[0]
        self.all_mels = torch.zeros((self.__len__(),*first_input.shape))
        self.all_mels[0] = self.normalize(first_input.unsqueeze(0))
        for idx in range(1,self.__len__(),64):
            pl = 64 if self.__len__() - idx >= 64 else self.__len__() - idx
            inputs_idx = list(range(idx,idx+pl))
            self.all_mels[idx:idx+len(inputs_idx)] = self.normalize(self.get_mel(inputs_idx))
        print('Shape of all the mel inputs',self.all_mels.shape)
        self.all_mels = self.all_mels.permute(0,2,1)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        if self.create_all_mels:
            X = self.all_mels[index]
        else:
            X = self.get_mel([index])[0].permute(1,0)
            mean = torch.mean(X,[0,1],True)
            std = torch.std(X,[0,1],True,True)
            X = (X - mean)/(std + 1e-10)
        y = self.labels[ID]
        return X,y
    
    def __len__(self):
        return len(self.list_IDs)
    
