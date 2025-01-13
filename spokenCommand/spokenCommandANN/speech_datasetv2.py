import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torchaudio
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchvision import transforms
import random
import numpy as np

LABELS = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


class GSpeechCommands(Dataset):
    def __init__(self, root, split_name, v2=True,transform=None, target_transform=None, download=True,augment=False):

        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = SPEECHCOMMANDS(root,download=download, subset=split_name)
        self.dim = 16000
        self.spec_aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=40)
        self.augment = augment
        
        
    def __len__(self):
        return len(self.dataset)

    def pad(self,tensor):
        if tensor.size(0) >= self.dim: return tensor[:,-self.dim:]
        else: 
            new_tensor = torch.zeros((tensor.size(0),self.dim))
            new_tensor[:,-tensor.size(1):] = tensor
            return new_tensor

    def __getitem__(self, index):
        waveform, _,label,_,_ = self.dataset[index]

        waveform = self.pad(waveform)
        mel = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=40)
        target = torch.tensor(LABELS.index(label))

        if self.augment:
            mel = self.spec_aug(mel)
            
        return mel, target
    
def get_dataloaders(root,args):

    train_dataset = GSpeechCommands(root, 'training',augment=args.augment)
    valid_dataset = GSpeechCommands(root, 'validation')
    test_dataset = GSpeechCommands(root, 'testing')

    g = torch.Generator()
    g.manual_seed(0)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, generator=g, worker_init_fn=seed_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    return train_loader, valid_loader, test_loader

