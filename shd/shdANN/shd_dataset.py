import torch
import librosa
import torchaudio
import soundfile
import os
import numpy as np
import tables
from brian2 import *
from loihi_equations import *
from joblib import Parallel, delayed
prefs.codegen.target = 'numpy'
from scipy.ndimage import gaussian_filter1d
import scipy 
from scipy import signal
import random
import torchaudio_augmentations
from torchaudio_augmentations import ComposeMany
from torchaudio_augmentations import Gain
from torchaudio_augmentations import Noise
from torchaudio_augmentations import PolarityInversion
from torchaudio_augmentations import RandomApply
from torchaudio_augmentations import Reverb


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



class SHD_MelDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_dataset,is_train=True, hop_length=128, n_fft=1024, n_mels=80,augment=False,min_max_norm=False):
        super().__init__()
        self.min_max_norm = min_max_norm
        self.is_train = is_train
        self.partition_txt = os.path.join(path_to_dataset,f"{'train' if is_train else 'test'}_filenames.txt")
        self.data = []
        self.labels = []
        self.tasks = []
        self.audios,self.labels = self.load_data()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = 40
        self.f_max = self.dim/2
        self.augment = augment
        self.mel_basis = torch.from_numpy(librosa.filters.mel(sr=48000,
                                                              n_fft=self.n_fft,
                                                              n_mels=self.n_mels,
                                                              htk=False,
                                                              norm=1))
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft,hop_length=(hop_length),win_length=self.n_fft,power=2,center=True)
        self.create_mels()

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.all_mels[idx], self.labels[idx]
    
    def load_data(self):
        with open(self.partition_txt, 'r') as file:
            file_paths = file.readlines()
        directory = os.path.join(os.path.dirname(self.partition_txt),"audio")
        file_paths = [os.path.join(directory, f.strip()) for f in file_paths]
        audios = []
        print("Length of the dataset: ", len(file_paths))
        audios_length = []
        labels = []
        i = 0
        for f in file_paths:
            audios.append(soundfile.read(f)[0])
            audios_length.append(audios[-1].shape[0])
            filename = os.path.basename(f)
            is_english = "english" in filename 
            digit = int(filename.split("digit-")[1].split(".")[0])
            labels.append(digit + (10 if is_english else 0))
            #if i > 0 and i % 100 == 0:
            #    break
            i += 1
        print("Loaded all the audios")
        self.dim = int(np.percentile(audios_length, 90))
        self.length = len(audios)
        return audios,labels
    
    def create_mels(self):
        print('Creating the mel dataset')
        first_input = self.get_mel([0])[0]
        self.all_mels = torch.zeros((self.__len__(),*first_input.shape))
        self.all_mels[0] = self.normalize(first_input.unsqueeze(0))
        for idx in range(1,self.__len__(),64):
            pl = 64 if self.__len__() - idx >= 64 else self.__len__() - idx
            inputs_idx = list(range(idx,idx+pl))
            self.all_mels[idx:idx+pl] = self.get_mel(inputs_idx)
        self.all_mels= self.normalize(self.all_mels)
        print('Shape of all the mel inputs',self.all_mels.shape)
        self.all_mels = self.all_mels.permute(0,2,1)

    def normalize(self,x):
        if self.min_max_norm:
            x = (x - torch.min(x,1,keepdim=True).values)/(torch.max(x,1,keepdim=True).values - torch.min(x,1,keepdim=True).values + 1e-10)
        else:
            x = (x - torch.mean(x,[1,2],True))/(torch.std(x,[1,2],True,True) + 1e-10)
        return x
    
    def melspec(self,x):
        new_spec = self.spec(x)
        new_mel = torch.matmul(self.mel_basis,new_spec)
        new_mel = torch.sqrt(new_mel)
        new_mel = amplitude_to_decibel(new_mel)
        #pads 0
        new_mel = new_mel[...,1:]
        return new_mel
    
    def cut_audio(self,curX):
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
        return X
    
    def get_mel(self,input_indices):
        torch_inputs = torch.empty(len(input_indices),self.dim)
        for i,input_idx in enumerate(input_indices):
            curX = torch.from_numpy(self.audios[input_idx]).float()
            torch_inputs[i] = self.cut_audio(curX)
            
        torch_inputs = self.melspec(torch_inputs)
        return torch_inputs
    
    
def binary_image_readout(times,units,dt = 1e-3):
    img = []
    N = int(1/dt)
    for i in range(N):
        idxs = np.argwhere(times<=i*dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700-vals] = 1
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
        
    return np.array(img)

def generate_dataset(file_name,dt=1e-3,number_of_samples=500):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    print("Number of samples: ",len(times))
    X = []
    y = []
    number_of_samples = min(number_of_samples,len(times)) if number_of_samples > 0 else len(times) 
    for i in range(number_of_samples):
        tmp = binary_image_readout(times[i], units[i],dt=dt)
        X.append(tmp)
        y.append(labels[i])
    
    fileh.close()
    return np.array(X),np.array(y)

# Functions taken from https://github.com/byin-cwi/Efficient-spiking-networks

class SHD_ConvertedDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_dataset,is_train=True,tau=0.0555851360941796404,dt=4e-3,number_of_samples=200,data_scaling=1.0,normalize=False,augment_data=False,use_gaussian_filter=False,convolve_spikes=False):
        super().__init__()
        self.tau = tau
        self.dt = dt
        self.data_scaling = data_scaling
        self.path_to_dataset = path_to_dataset
        self.number_of_samples = number_of_samples
        self.is_train = is_train
        if self.is_train:
            self.filename = "shd_train.h5"
        else:
            self.filename = "shd_test.h5"
        self.convolve_spikes = convolve_spikes
        self.augment_data = augment_data
        self.do_normalize = normalize
        if not use_gaussian_filter:
            self.get_data_brian2()
        else:
            self.get_data_gaussian()
            
        if self.augment_data and self.is_train:
            transforms = [
                torchaudio_augmentations.Noise(),
                torchaudio_augmentations.RandomApply([torchaudio_augmentations.Delay(sample_rate=1000,volume_factor=1,min_delay=30,max_delay=100,delay_interval=10)], p=0.7),
                torchaudio_augmentations.RandomApply([torchaudio_augmentations.Gain()], p=0.3),
                torchaudio_augmentations.RandomApply([lambda x: torch.roll(x,random.randint(0,20),0)], p=0.5),
            ]
            self.transforms = torchaudio_augmentations.Compose(transforms=transforms)

    def get_data_gaussian(self):
        converted_filename = self.filename.split(".h5")[0]+"_{:.4f}_{:4f}_converted_gaussian.pth".format(self.tau,self.dt)
        if os.path.isfile(os.path.join(self.path_to_dataset,converted_filename)):
            print("Converted file already exists")
            self.inputs = torch.load(os.path.join(self.path_to_dataset,converted_filename)).permute(0,2,1)
            self.y = torch.load(os.path.join(self.path_to_dataset,self.filename.split(".h5")[0]+"_{:.4f}_{:4f}_labels.pth".format(self.tau,self.dt)))
            print("Loaded the converted file")
            print("Shape of the converted file",self.inputs.shape)
        else:
            spikes, self.y = generate_dataset(os.path.join(self.path_to_dataset,self.filename),dt=self.dt,number_of_samples=self.number_of_samples)
            self.inputs = self.gaussian_filter(spikes)
            self.inputs = torch.from_numpy(self.inputs).float()
            self.y = torch.from_numpy(self.y.astype(int)).long()
            if self.do_normalize:
                self.inputs = self.normalize(self.inputs)
            torch.save(self.inputs,os.path.join(self.path_to_dataset,converted_filename))
            torch.save(self.y,os.path.join(self.path_to_dataset,self.filename.split(".h5")[0]+"_{:.4f}_{:4f}_labels.pth".format(self.tau,self.dt)))

    def gaussian_filter(self,spikes):
        gaussed = []
        for i in range(spikes.shape[0]):
            gaussed.append(gaussian_filter1d(spikes[i],sigma=2,axis=0))
        gaussed = np.array(gaussed)
        return gaussed
    def get_data_brian2(self):
        converted_filename = self.filename.split(".h5")[0]+"_{:.4f}_{:4f}_{}_converted_brian2.pth".format(self.tau,self.dt,self.convolve_spikes)
        if os.path.isfile(os.path.join(self.path_to_dataset,converted_filename)):
            print("Converted file already exists")
            self.inputs = torch.load(os.path.join(self.path_to_dataset,converted_filename))
            self.y = torch.load(os.path.join(self.path_to_dataset,self.filename.split(".h5")[0]+"_{:.4f}_{:4f}_labels.pth".format(self.tau,self.dt)))
            print("Loaded the converted file")
            print("Shape of the converted file",self.inputs.shape)
        else:
            spikes, self.y = generate_dataset(os.path.join(self.path_to_dataset,self.filename),dt=self.dt,number_of_samples=self.number_of_samples) 
            self.inputs = self.brian2_convert_to_analog(spikes)
            del spikes
            self.inputs = self.inputs*self.data_scaling
            self.inputs = torch.from_numpy(self.inputs).float()
            self.y = torch.from_numpy(self.y.astype(int)).long()
            if self.do_normalize:
                self.inputs = self.normalize(self.inputs)
            torch.save(self.inputs,os.path.join(self.path_to_dataset,converted_filename))
            torch.save(self.y,os.path.join(self.path_to_dataset,self.filename.split(".h5")[0]+"_{:.4f}_{:4f}_labels.pth".format(self.tau,self.dt)))
        

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        label = self.y[idx]
        if self.augment_data and self.is_train:
            inputs = self.transforms(inputs)
        return inputs.T,label.long()
    
    def normalize(self,x):
        mean = torch.mean(x,[1,2],True)
        std = torch.std(x,[1,2],True,True)
        x = (x - mean)/(std + 1e-10)
        return x

    def brian2_convert_to_analog(self, spikes):
        def run_brian(ip_idx):
            net = Network()

            ip = spikes[ip_idx]
            if self.convolve_spikes:
                ip = scipy.ndimage.convolve1d(ip, weights=signal.windows.gaussian(10, std=2))[:,::5]

            num_neurons =  ip.shape[1]
            tau_snn = self.tau/self.dt
            
            defaultclock.dt = self.dt*second
            input_groups = NeuronGroup(num_neurons, **loihi_eq_dict)
            neuron_params = {}
            neuron_params['tau_v'] = tau_snn
            neuron_params['tau_in'] = tau_snn
            neuron_params['tau_ip'] = tau_snn
            neuron_params['tau_fb'] = tau_snn
            neuron_params['gain'] = tau_snn
            neuron_params['vThMant'] = 0
            neuron_params['ref_p'] = 0

            Iin_snn = 10
            feedback_connection =  Synapses(input_groups, input_groups, **loihi_syn_dict)
            feedback_connection.connect(condition='i==j')
            feedback_connection.weight[:] = - Iin_snn / (64 * tau_snn)#self.fb_weight
            feedback_connection.w_factor[:] = 1

            set_params(input_groups, neuron_params)

            b_ip = TimedArray(ip,dt=self.dt*second)
            run_reg = input_groups.run_regularly('''i_in += b_ip(t,i)/tau_snn**2''', dt=self.dt*second)  
            input_sine_mon  = StateMonitor(input_groups, ['i_ip'], record=[i for i in range(num_neurons)])

            simulation_time = (spikes[ip_idx].shape[0])*self.dt*second
            net.add(input_groups)
            net.add(input_sine_mon)
            net.add(feedback_connection)
            net.add(run_reg)
            net.run(simulation_time)
            return input_sine_mon.i_ip

        print(f"Converting {spikes.shape[0]} files to analog")
        analogs = Parallel(n_jobs=4)(delayed(run_brian)(int(test_num)) for test_num in range(len(spikes)))
        analogs = np.array(analogs)
        print("Data set shape",analogs.shape)
        return analogs
        
