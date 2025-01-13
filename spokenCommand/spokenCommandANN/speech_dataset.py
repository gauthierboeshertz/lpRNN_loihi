import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import torchaudio
import SpeechDownloader  
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)

    
class SpeechDataset(Dataset):
    def __init__(self,
                  list_IDs,
                  labels,
                  dim=16000,
                  train_or_val='train',
                  augment=False):
        self.dim = dim
        self.labels = labels
        self.list_IDs = list_IDs
        self.train_or_val = train_or_val
        self.augment = augment
        if self.augment:
            self.spec_aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=40)

    def pad_input(self,inputs):
        X = torch.zeros((self.dim)).float()
        if inputs.shape[0] == self.dim:
            X = inputs
        elif inputs.shape[0] > self.dim: #bigger
            X = inputs[-self.dim:]
        else: 
            X[-inputs.shape[0]:] = inputs
        return X
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        curX = torch.from_numpy(np.load(ID)).float()
        X = self.pad_input(curX)
        X = X.unsqueeze(0)
        X = torchaudio.compliance.kaldi.fbank(X, num_mel_bins=40)
        if self.augment:
            X = self.spec_aug(X)
        y = self.labels[ID]
        return X,y
        
    def mel_spec(x):
        return torchaudio.compliance.kaldi.fbank(x, num_mel_bins=40)

    def __len__(self):
        return len(self.list_IDs)
    

def get_dataloaders(root,args):
    gscInfo, nCategs = SpeechDownloader.PrepareGoogleSpeechCmd(version=2 if args.v2 else 1, task = '35word' if args.task_n_words == 36 else '12cmd', base_path_prefix=root)
    trainDs   = SpeechDataset(gscInfo['train']['files']
                                        ,gscInfo['train']['labels'],train_or_val='train',augment=args.augment)
    valDs   = SpeechDataset(gscInfo['val']['files']
                                        ,gscInfo['val']['labels'],train_or_val='val')
    test_ds   = SpeechDataset(gscInfo['test']['files'],gscInfo['test']['labels'],train_or_val='val')

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(trainDs,batch_size=args.batch_size,num_workers=8,drop_last=False,shuffle=True,pin_memory=True,generator=g,worker_init_fn=seed_worker)
    val_loader = DataLoader(valDs,batch_size=args.batch_size,shuffle=False,num_workers=8,pin_memory=False,generator=g)
    test_loader = DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=8,pin_memory=False,generator=g)
    return train_loader, val_loader, test_loader