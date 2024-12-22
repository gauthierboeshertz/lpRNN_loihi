
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
from layers import quantize_weight,jit_lpRNN,jit_lpLinear,Dense
import speech_dataset
import SpeechDownloader
from speech_datasetv2 import GSC_dataloaders

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

    vers_str = 'v2' if args.v2 else 'v1'
    if args.wth < 0:
        model_path = './weights'+'/' +  ("lstm" if args.use_lstm else "QlpRNN")  + "_" +str(args.lpff_size)+'_' + str(args.nlayers)+ '_' + str(args.nunits) +'intbits_'+str(args.int_bits)+'frac_bits'+ str(args.frac_bits)+str(args.ret_ratio) +'_'+str(args.task_n_words)+("_augment" if args.augment else "")+'.pth'
    else:
        if args.use_lstm:
            model_path = './weights'+'/lstm_' + str(args.nlayers)+ '_' + str(args.nunits) +'_nWTH_'+str(args.wth)+'_actreg'+str(args.activity_reg_weight) +"_"+vers_str+'_'+str(args.task_n_words)+'.pth'
        else:
            data_name = f"hop_{args.hop_length}_win_{args.win_length}_n_ftt_{args.n_fft}"+(f"n_mels_{args.n_mels}" if args.n_mels != 80 else "")
            model_path = './weights'+'/' +("small_" if args.small_net else "")+args.model +"_" +str(args.lpff_size)+ '_' + str(args.nlayers)+ '_' + str(args.nunits) +'intbits_'+str(args.int_bits)+'frac_bits'+ str(args.frac_bits)+str(args.ret_ratio) +'_nWTH_'+str(args.wth) +'_'+str(args.task_n_words)+("_augment" if args.augment else "")+data_name+("no_bias" if  args.no_bias else "")+f'seed{args.seed}.pth'

    print("MODEL PATH",model_path)
    use_old_ds = (args.task_n_words == 12 or args.task_n_words == 36)
    if use_old_ds:
        if args.task_n_words == 36:
            task = '35word'
            version = 1 
        if args.task_n_words == 12:
            task = '12cmd'
            version = 2
        
        gscInfo, nCategs = SpeechDownloader.PrepareGoogleSpeechCmd(version=2 if args.v2 else 1, task = '35word' if args.task_n_words == 36 else '12cmd', base_path_prefix="data/")
        test_ds   = speech_dataset.SpeechDataset(gscInfo['test']['files']
                                            ,gscInfo['test']['labels'],
                                            create_all_mels=False,
                                            train_or_val='val',
                                            hop_length=args.hop_length,
                                            n_fft=args.n_fft)
        
        test_loader = DataLoader(test_ds,batch_size=args.batch_size,num_workers=8,drop_last=False,shuffle=False,pin_memory=False)

    else:
        _,_, test_loader = GSC_dataloaders('data/',args=args)
    print(model_path)
    sr = 16000
    n_classes = 35 if args.v2 and args.task_n_words == 35 else 12 if args.v2 else 36
    model = models.small_lprnn_net(n_classes,
                        lpff_size=args.lpff_size,
                        input_size=args.n_mels,
                        nunits=args.nunits,
                        nlayers=args.nlayers,
                        load_mel=False,
                        use_lstm=args.use_lstm,
                        use_bias=not args.no_bias,
                        int_bits=args.int_bits,
                        frac_bits=args.frac_bits,
                        ret_ratio=args.ret_ratio,
                        dev='cuda')

    model.load_state_dict(torch.load(model_path))


    model = model.to(args.device)
    model.eval()


    data_path = '../spokenCommandSNN/data/test_data'+'_'+str(args.task_n_words)+data_name+'.npy'
    label_path = '../spokenCommandSNN/data/test_labels'+'_'+str(args.task_n_words)+data_name+'.npy'
    model_save_path = '../spokenCommandSNN/data/'+ 'QlpRNN_'+str(args.lpff_size)+"_"+str(args.nlayers)+'_'+str(args.nunits)+'_'+str(args.ret_ratio)+'_'+str(args.frac_bits)+'_nWTH_'+str(args.wth) +'_'+str(args.task_n_words)+("_augment" if args.augment else "")+("_no_bias_" if args.no_bias else "" )+data_name+'.pth'
    save_weights(model,model_save_path)
    test_and_save_data(model,test_loader,nn.CrossEntropyLoss(),data_path,label_path)
    return 

def save_weights(model,model_path):
    q_model_dict = dict()
    q_model_dict['order'] = []
    for l_idx,layer in (enumerate(model.modules())):
        if isinstance(layer, jit_lpRNN):
            if args.no_bias:
                w = [quantize_weight(layer.linear.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),quantize_weight(layer.linear_hh.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),torch.zeros_like(layer.linear.bias.data).cpu().numpy()]
            else:
                w = [quantize_weight(layer.linear.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),quantize_weight(layer.linear_hh.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),quantize_weight(layer.linear.bias.data,args.int_bits,args.frac_bits).cpu().numpy()]
            q_model_dict[layer.name] = w
            q_model_dict['order'].append(layer.name)
        elif isinstance(layer,(Dense,jit_lpLinear)):
            if args.no_bias:
                w = [quantize_weight(layer.linear.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),torch.zeros_like(layer.linear.bias.data).cpu().numpy()]
            else:
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
    #preds = torch.mean(outputs[:,outputs.shape[1]//2:],axis=1)
    loss =  criterion(outputs,targets)
    acc = get_accuracy(outputs,targets)
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
    print('TEST ACCURACY',meanacc)
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
    parser.add_argument('--augment', action='store_true')

    parser.add_argument('--wth', type=int, default=-1,
                        help='winning ticket hypothesis iteration')
    parser.add_argument('--activity_reg_weight',type=float,default=-1,#0.000003
                        help='winning ticket hypothesis number of iterations')
    parser.add_argument('--task_n_words',type=int,default=36,
                        help='gsc task')
    parser.add_argument('--small_net',action='store_true')
    parser.add_argument('--v2',action="store_true")
    parser.add_argument('--hop_length',type=int,default=128)
    parser.add_argument('--n_mels',type=int,default=80)
    parser.add_argument('--lpff_size',type=int,default=128)
    parser.add_argument('--minmax_axis',type=int,default=-1)
    parser.add_argument('--ret_ratio',type=float,default=0.8)
    parser.add_argument('--n_fft',type=int,default=1024)
    parser.add_argument('--win_length',type=int,default=1024)
    parser.add_argument('--no_bias',action='store_true')
    parser.add_argument('--seed',type=int,default=0)

    args = parser.parse_args()
    print(args)
    save_data(args)

