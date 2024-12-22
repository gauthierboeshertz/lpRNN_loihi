
# coding: utf-8
# # Neuromorphic speech command recognition model
# We reuse an existing code sourced from  https://github.com/manuvn/SpeechCmdRecognition.git.
# The original code provided us with a good and readily usable infrastructure to plug in the proposed low pass RNN models.

# Following stderr write disables "Using tensorflow backend" messages
import os, sys
import pickle

import math
import argparse
from datetime import datetime
import torch
#import speech_dataset
import signal
import numpy as np
import models
#import SpeechDownloader
from torch.utils.data import DataLoader
import torch.nn as nn
from layers import *
import shd_dataset

def count_pruned_weights(mask):
    total_pruned = 0
    count = 0
    for name in mask.keys():
        npruned = (mask[name] == 0).sum().cpu().data.numpy()
        nweight = mask[name].numel()
        print('Weight',name,'have ',(npruned/nweight) *100,' % of its weight pruned')
        total_pruned += npruned
        count += mask[name].numel()
    print('In total ',total_pruned/count *100,'% of the weights are pruned')


def save_data(args):
    ret_ratio = args.ret_ratio
    print('retention ratio is :',ret_ratio)
    
    model_inp = (f"mel_{args.n_mels}_hop_length_{args.hop_length}") if args.use_mels else "spikes" + f"_conv_spikes_{args.convolve_spikes}_dt_{args.dt}_tau_{args.tau}"
    model_name = "lstm" if args.use_lstm else f"{'no_lpff_' if args.no_first_lpff else ''}QlpRNN" + (f"_lpff_size_{args.lpff_size}")  +f"_{args.nlayers}_{args.nunits}"+f"_ret_{args.ret_ratio}"

    if args.wth < 0:
        model_path = f"./weights/{model_name}_{model_inp}_{args.nlayers}_{args.nunits}intbits_{args.int_bits}frac_bits{args.frac_bits}_{ret_ratio}.pth"
    else:
        if args.use_lstm:
            model_path = f"./weights/lstm_{model_inp}_{args.nlayers}_{args.nunits}_nWTH_{args.wth}.pth"
        else:
            model_path = f"./weights/{model_name}_{model_inp}_nWTH_{args.wth}.pth"
    # Speech Data Generator 
    print(model_path)
    print(model_path)
    input_size = (80) if args.use_mels else 140 if args.convolve_spikes else 700
    model = models.small_lprnn_net(20,
                        input_size=input_size,
                        int_bits=args.int_bits,
                        frac_bits=args.frac_bits,
                        nunits=args.nunits,
                        nlayers=args.nlayers,
                        no_first_lpff=args.no_first_lpff,     
                        lpff_size=args.lpff_size,  
                        ret_ratio=ret_ratio,
                        dev=args.device,
                        use_lstm=args.use_lstm)
    model.load_state_dict(torch.load(model_path))

    if args.use_mels:
        testDs = shd_dataset.SHD_MelDataset("data/hd_audio",is_train=False,hop_length=args.hop_length)
    else:
        testDs = shd_dataset.SHD_ConvertedDataset("data/hd_audio",is_train=True,number_of_samples=-1,data_scaling=args.data_scaling,normalize=True)

    print(model)
    test_loader = DataLoader(testDs,batch_size=args.batch_size,num_workers=3,drop_last=False,shuffle=False,pin_memory=False)

    model = model.to(args.device)
    model.eval()
    data_path = '../shdSNN/data/test_data.npy'
    label_path = '../shdSNN/data/test_labels.npy'
    model_save_path = '../shdSNN/data/'+f'QlpRNN_{model_inp}_'+str(args.nlayers)+'_'+str(args.nunits)+'_'+str(ret_ratio)+'_'+str(args.frac_bits)+'_nWTH_'+str(args.wth) +'.pth'
    print("Not saving weieghts")
    save_weights(model,model_save_path)
    test_and_save_data(model,test_loader,nn.CrossEntropyLoss(),data_path,label_path)
    return 

def save_weights(model,model_path):
  q_model_dict = dict()
  q_model_dict['order'] = []
  for l_idx,layer in (enumerate(model.modules())):
    if isinstance(layer, jit_lpRNN):
      w = [quantize_weight(layer.linear.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),quantize_weight(layer.linear_hh.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),quantize_weight(layer.linear.bias.data,args.int_bits,3).cpu().numpy()]
      q_model_dict[layer.name] = w
      q_model_dict['order'].append(layer.name)
    elif isinstance(layer,(Dense,jit_lpLinear)):
      w = [quantize_weight(layer.linear.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),quantize_weight(layer.linear.bias.data,args.int_bits,args.frac_bits).cpu().numpy()]
      q_model_dict[layer.name] = w
      q_model_dict['order'].append(layer.name)

    new_mask = dict()
    for name, weight in model.named_parameters():
        if 'bias' not in name and 'ratio' not in name:
            new_mask[name] = quantize_weight(weight,args.int_bits,args.frac_bits) != 0
      
  count_pruned_weights(new_mask)

  with open(model_path, 'wb') as handle: 
    pickle.dump(q_model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print('saved weights')
    
def get_accuracy( output, target):
  output = torch.argmax(output, dim=-1)
  accuracy = torch.sum(output == target)
  accuracy = accuracy.to(float)
  return accuracy

def get_loss_acc( outputs, targets,criterion):
    preds = torch.mean(outputs[:,outputs.shape[1]//2:],axis=1)
    loss =  criterion(preds,targets)
    acc = get_accuracy(preds,targets)
    return loss,acc.cpu().data.numpy()

def test_and_save_data(model,loader,criterion,data_path,label_path):
    print('testing and saving data')
    model.eval()
    val_acc, val_loss = 0,0

    for batch_idx,batch in enumerate((loader)):
        data = batch[0].to(args.device)
        input_shape = data.shape[1:][::-1]
        break

    all_data = torch.zeros((len(loader.dataset),*input_shape)).to(args.device)
    all_labels =  torch.zeros((len(loader.dataset),)).to(args.device)
    with torch.no_grad():
        for batch_idx,batch in enumerate((loader)):
            data = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            y = model(data.clone())
            data = data.permute(0,2,1)
            all_data[batch_idx*loader.batch_size:loader.batch_size*(batch_idx+1)] = data
            all_labels[batch_idx*loader.batch_size:loader.batch_size*(batch_idx+1)] = labels
            temp_val_loss,temp_acc = get_loss_acc(y,labels,criterion)
            val_loss += temp_val_loss.cpu().data.numpy()
            val_acc += temp_acc
    meanloss = val_loss/len(loader)
    meanacc = val_acc/(len(loader.dataset))
    print('test loss:',meanloss,'TEST ACCURACY',meanacc)
    np.save(data_path,all_data.cpu().numpy())
    np.save(label_path,all_labels.cpu().numpy())
    print('done with saving the test data')
    return model, meanloss,meanacc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Network settings
    
    parser.add_argument('--model', type=str, default='QlpRNN',
                        help='RNN type to use. lpRNN, lrlpRNN, lpLSM, CuDNNLSTM, SimpleRNN, QlpRNN')
    parser.add_argument('--nunits', type=int, default=256,
                        help='number of hidden layer units')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of hidden layer units')
    parser.add_argument('--use_bn', action='store_true')

    parser.add_argument('--ts_ann', type=float, default=0.008,
                        help='NUmber of integer weight bits')
    parser.add_argument('--tau', type=float, default=0.035851360941796404,##tau for a retention ratio of 0.8
                        help='NUmber of integer weight bits')#               and timestep of 1/125

    parser.add_argument('--int_bits', type=int, default=32,
                        help='NUmber of integer weight bits')
    parser.add_argument('--frac_bits', type=int, default=3,
                        help='NUmber of fractional weight bits')
    parser.add_argument('--use_lstm', action='store_true',
                        help='use dropout in the dense layers')
    parser.add_argument('--batch_size', type=int,default=64,
                        help='use dropout in the dense layers')

    parser.add_argument('--use_bias', type=bool, default=True,)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--augmented', action='store_true')

    parser.add_argument('--wth', type=int, default=-1,
                        help='winning ticket hypothesis iteration')
    parser.add_argument('--task_n_words',type=int,default=36,
                        help='gsc task')
    parser.add_argument('--small_net',action='store_true')
    parser.add_argument('--use_mels',action='store_true')
    parser.add_argument('--data_scaling',type=float,default=1.0)
    parser.add_argument('--no_first_lpff',action='store_true')
    parser.add_argument('--hop_length',type=int,default=128)
    parser.add_argument('--dt',type=float,default=128)
    parser.add_argument('--min_max_norm',action="store_true")
    parser.add_argument('--do_norm',action="store_true")
    parser.add_argument('--lpff_size',default=128,type=int)
    parser.add_argument('--n_mels',default=80,type=int)
    parser.add_argument('--train_ret_ratio',action="store_true")
    parser.add_argument('--convolve_spikes',action="store_true")
    parser.add_argument('--ret_ratio',type=float,default=0.8)

    args = parser.parse_args()
    print(args)
    save_data(args)

