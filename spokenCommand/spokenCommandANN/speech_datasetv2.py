import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchvision import transforms
import librosa
from torch import nn, Tensor
from torch.distributions import Uniform
from torch.nn import functional as F


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

def build_my_transform(args,is_train=False):
    
    sample_rate = 16000
    f_min = 40
    f_max = sample_rate//2
    
    mel_basis = torch.from_numpy(librosa.filters.mel(sr=sample_rate//2, n_fft=args.n_fft, n_mels=80,
                            fmin=f_min, fmax=f_max,
                            htk=False, norm=1))
    spec = torchaudio.transforms.Spectrogram(n_fft=args.n_fft,hop_length=args.hop_length,win_length=args.win_length,power=2,center=True)
    
    spec_aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)
    def mel_spec(x):
        new_spec = spec(x)
        if args.augment and is_train:
            new_spec = spec_aug(new_spec)
        new_mel = torch.matmul(mel_basis,new_spec)
        new_mel = torch.sqrt(new_mel)
        new_mel = amplitude_to_decibel(new_mel)
        new_mel = new_mel[...,1:]
        return new_mel

    def pad(tensor):
        if tensor.size(0) > sample_rate: return tensor[-sample_rate:]
        else: return F.pad(tensor, (0, sample_rate - tensor.size(1)))

    t = [pad]

    if args.augment and is_train:       
        """
        augs = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            Shift(p=0.5),
            ])
        
        def augzer(x):
            rr = torch.from_numpy(augs(samples=x.numpy(), sample_rate=sample_rate))
            return rr
        
        t.append(augzer)
        """
    t.append(mel_spec)
        
    def mean_std(x):
        mean = torch.mean(x,[1,2],True)
        std = torch.std(x,[1,2],True,True)
        return (x - mean)/(std + 1e-10)

    t.append(mean_std)
    
    return transforms.Compose(t)


labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

target_transform = lambda word : torch.tensor(labels.index(word))


def GSC_dataloaders(root,args):

    train_dataset = GSpeechCommands(root, 'training', transform=build_my_transform(args,is_train=True), target_transform=target_transform)
    valid_dataset = GSpeechCommands(root, 'validation', transform=build_my_transform(args,is_train=False), target_transform=target_transform)
    test_dataset = GSpeechCommands(root, 'testing', transform=build_my_transform(args,is_train=False), target_transform=target_transform)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    return train_loader, valid_loader, test_loader

class GSpeechCommands(Dataset):
    def __init__(self, root, split_name, v2=True,transform=None, target_transform=None, download=True):

        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = SPEECHCOMMANDS(root,download=download, subset=split_name)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        waveform, _,label,_,_ = self.dataset.__getitem__(index)

        if self.transform is not None:
            waveform = self.transform(waveform).squeeze().t()

        target = label

        if self.target_transform is not None:
            target = self.target_transform(target)

        return waveform, target