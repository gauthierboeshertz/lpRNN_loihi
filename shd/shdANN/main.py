
# coding: utf-8
# # Neuromorphic speech command recognition model
# We reuse an existing code sourced from  https://github.com/manuvn/SpeechCmdRecognition.git.
# The original code provided us with a good and readily usable infrastructure to plug in the proposed low pass RNN models.

# Following stderr write disables "Using tensorflow backend" messages
import warnings 
warnings.filterwarnings(action='once')

from layers.q_utils import quantize_weight,rsetattr,rgetattr
import os
import math
import argparse
import torch
import numpy as np
import shd_dataset
import models
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import torch.nn.functional as F

from layers import Dense, jit_lpRNN,jit_lpLinear
import random
os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
def seed_worker(worker_id):
    np.random.seed(1)
    random.seed(1)

def adjust_learning_rate(optimizer, epoch):
  epochs_drop = 7
  new_lr = args.lr *  math.pow(0.7,math.floor((epoch)/epochs_drop))
  for param_group in optimizer.param_groups:
      if new_lr > 4e-6:
          print('learning rate adjusted to :',new_lr)
          param_group['lr'] = new_lr

def count_pruned_weights(mask):
    total_pruned = 0
    count = 0
    print('='*20)
    for name in mask.keys():
        npruned = (mask[name] == 0).sum().cpu().data.numpy()
        nweight = mask[name].numel()
        print('Weight',name,'have ',(npruned/nweight) *100,' % of its weight pruned')
        total_pruned += npruned
        count += mask[name].numel()
    print('In total ',total_pruned/count *100,'% of the weights are pruned')
    print('='*20)


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

def run_test_case(args,winning_ticket_iter=-1,first_initialization=None, wth_mask=None):
    print('Tau is ',args.tau)
    print('Timestep is ',args.ts_ann)
    ret_ratio = args.ret_ratio
    print('retention ratio is :',ret_ratio)

    model_inp = (f"mel_{args.n_mels}_hop_length_{args.hop_length}") if args.use_mels else "spikes" + f"_conv_spikes_{args.convolve_spikes}_dt_{args.dt}_tau_{args.tau}"
    model_name = "lstm" if args.use_lstm else f"{'no_lpff_' if args.no_first_lpff else ''}QlpRNN" + (f"_lpff_size_{args.lpff_size}")  +f"_{args.nlayers}_{args.nunits}"+f"_ret_{args.ret_ratio}"

    if winning_ticket_iter < 0:
        model_path = f"./models/{model_name}_{model_inp}_{args.nlayers}_{args.nunits}intbits_{args.int_bits}frac_bits{args.frac_bits}_{ret_ratio}.pth"
    else:
        if args.use_lstm:
            model_path = f"./models/lstm_{model_inp}_{args.nlayers}_{args.nunits}_nWTH_{winning_ticket_iter}.pth"
        else:
            model_path = f"./models/{model_name}_{model_inp}_nWTH_{winning_ticket_iter}.pth"

    # Speech Data Generator 
    print("model path",model_path)

    criterion = nn.CrossEntropyLoss() 

    input_size = args.n_mels if args.use_mels else 140 if args.convolve_spikes else 700
    model = models.small_lprnn_net(20,
                                input_size=input_size,
                                nunits=args.nunits,
                                nlayers=args.nlayers,
                                no_first_lpff=args.no_first_lpff,     
                                lpff_size=args.lpff_size,    
                                ret_ratio=args.ret_ratio,
                                use_lstm=args.use_lstm,
                                int_bits=args.int_bits,
                                frac_bits=args.frac_bits,
                                dev=DEVICE)
    previous_mask = None
    print(model)
    if winning_ticket_iter == 0:
        first_initialization = model.state_dict()
        previous_mask = {}
        for name, weight in model.named_parameters():
            if 'bias' not in name and 'ratio' not in name:
                previous_mask[name] = torch.ones_like(weight).bool()
    elif winning_ticket_iter > 0:
        model.load_state_dict(first_initialization)
        previous_mask = wth_mask

    model = model.to(DEVICE)
    if previous_mask is not None:
        count_pruned_weights(previous_mask)
        for name, weight in model.named_parameters():
            if 'bias' not in name and 'ratio' not in name:
                rsetattr(model,name+'.data',weight*previous_mask[name].to(DEVICE))
    
    if args.use_mels:
        train_ds = shd_dataset.SHD_MelDataset("data/hd_audio",is_train=True,hop_length=args.hop_length,min_max_norm=args.min_max_norm,n_mels=args.n_mels)
        val_ds = shd_dataset.SHD_MelDataset("data/hd_audio",is_train=False,hop_length=args.hop_length,min_max_norm=args.min_max_norm,n_mels=args.n_mels)
    else:
        train_ds = shd_dataset.SHD_ConvertedDataset("data/hd_audio",is_train=True,
                                                    tau=args.tau,
                                                    dt=args.dt,
                                                    number_of_samples=-1,
                                                    data_scaling=args.data_scaling,
                                                    normalize=True,
                                                    augment_data=args.augment_data,
                                                    use_gaussian_filter=args.use_gaussian_filter,
                                                    convolve_spikes=args.convolve_spikes)
        val_ds =  shd_dataset.SHD_ConvertedDataset("data/hd_audio",is_train=False,tau=args.tau,number_of_samples=-1,data_scaling=args.data_scaling,dt=args.dt, normalize=True,augment_data=False,use_gaussian_filter=args.use_gaussian_filter,convolve_spikes=args.convolve_spikes)
            
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_ds,batch_size=args.batch_size,num_workers=4,drop_last=False,shuffle=True,pin_memory=True,generator=g)
    val_loader = DataLoader(val_ds,batch_size=128,shuffle=False,num_workers=4,pin_memory=False,generator=g)

    optimizer = torch.optim.Adam(lr=args.lr,params = model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    print(optimizer)
    max_val_acc = -1
    for cur_epoch in range(args.nepochs):
        if (cur_epoch%args.print_freq==0):
            print('IN EPOCH :',cur_epoch)
        #adjust_learning_rate(optimizer, cur_epoch)
        model,val_loss, val_acc = train_epoch(model,optimizer,criterion,train_loader,val_loader,previous_mask,print_metrics=(cur_epoch%args.print_freq==0))
        if np.isnan(val_loss).any():
            print('NaN loss exiting early!')
            break
        if val_acc> max_val_acc:
            print('saving model to memory, current accuracy ',val_acc,'is better than previous',max_val_acc)
            max_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
        scheduler.step(val_loss)

    print('FINISHED TRAINING')
    model.load_state_dict(torch.load(model_path))
    if previous_mask is not None:
        new_mask = dict()
        for name, weight in model.named_parameters():
            if 'b' not in name and 'ratio' not in name:
                q_weight = quantize_weight(weight,args.int_bits,args.frac_bits)
                new_mask[name] = q_weight != 0
        count_pruned_weights(new_mask)

    if winning_ticket_iter >= 0:
        return max_val_acc, first_initialization, new_mask
    else:
        return max_val_acc


def get_accuracy( output, target):
  output = torch.argmax(output, dim=-1)
  accuracy = torch.sum(output == target)
  accuracy = accuracy.to(float) /output.shape[0] # batch first
  return accuracy*100

def get_loss_acc( outputs, targets,criterion):
    preds = torch.mean(outputs[:,outputs.shape[1]//2:],axis=1)
    loss =  criterion(preds,targets)
    acc = get_accuracy(preds,targets)
    return loss,acc.cpu().data.numpy()

def train_epoch(model,optimizer,criterion,train_dataloader,val_dataloader,previous_mask = None,print_metrics=False):
    total_acc, total_loss = 0, 0
    model.train()
    
    for train_batch_idx, train_batch in enumerate((train_dataloader)):
        
        if torch.any(train_batch[0] != train_batch[0]):
          print('nan in the data')
                        
        train_data = train_batch[0].to(DEVICE)
        train_labels = train_batch[1].to(DEVICE)


        optimizer.zero_grad()
        y = model(train_data)
        loss,acc = get_loss_acc(y,train_labels,criterion)


        loss.backward()
        # wth
        if previous_mask is not None:
            for name, weight in model.named_parameters():
                if ('bias' not in name and 'ratio' not in name) :
                    weight.grad.data = torch.where(previous_mask[name].to(DEVICE),weight.grad.data,torch.zeros_like(weight.grad.data).to(DEVICE))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()

        total_loss += loss.cpu().data.numpy()
        total_acc += acc
    
    _,val_loss,val_acc = validate(model,val_dataloader,F.cross_entropy)
    if print_metrics:
        print('training loss', total_loss/len(train_dataloader),'training accuracy', total_acc/(len(train_dataloader)))
        print('val loss:',val_loss,'val accuracy',val_acc)
    return model,val_loss,val_acc


def validate(model,loader,criterion):
    model.eval()
    val_acc, val_loss = 0,0
    with torch.no_grad():
        for batch_idx,batch in enumerate((loader)):
            data = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            y = model(data)
            temp_val_loss,temp_acc = get_loss_acc(y,labels,criterion)
            val_loss += temp_val_loss.cpu().data.numpy()
            val_acc += temp_acc
    meanloss = val_loss/len(loader)
    meanacc = val_acc/(len(loader))
    return model, meanloss,meanacc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='QlpRNN',
                        help='RNN type to use. lpRNN, lrlpRNN, lpLSM, CuDNNLSTM, SimpleRNN, QlpRNN')
    parser.add_argument('--nunits', type=int, default=256,
                        help='number of hidden layer units')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of hidden layer units')
    parser.add_argument('--ts_ann', type=float, default=0.008,
                        help='NUmber of integer weight bits')
    parser.add_argument('--int_bits', type=int, default=32,
                        help='NUmber of integer weight bits')
    parser.add_argument('--frac_bits', type=int, default=32,
                        help='NUmber of fractional weight bits')
    parser.add_argument('--use_lstm', action='store_true',
                        help='use dropout in the dense layers')
    parser.add_argument('--clipnorm', type=float, default=5,
                        help='Clip gradient norm for optimizer. 0=disabled')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--clipvalue', type=float, default=5,
                        help='Clip gradient value for optimizer. 0=disabled')
    parser.add_argument('--nepochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--wth', type=int, default=-1,
                        help='winning ticket hypothesis number of iterations')
    parser.add_argument('--resume_wth', type=int, default=0,
                        help='winning ticket hypothesis number of iterations')
    parser.add_argument('--task_n_words',type=int,default=36,
                        help='gsc task')
    parser.add_argument('--print_freq',type=int,default=10)
    parser.add_argument('--use_mels',action="store_true")
    parser.add_argument('--no_first_lpff',action="store_true")
    parser.add_argument('--data_scaling',type=float,default=1.0)
    parser.add_argument('--augment_data',action="store_true")
    parser.add_argument('--use_gaussian_filter',action="store_true")
    parser.add_argument('--tau',type=float,default=0.0555851360941796404,help="tau for the converted dataset")
    parser.add_argument('--convolve_spikes',action="store_true")
    parser.add_argument('--ret_ratio',type=float,default=0.8)
    parser.add_argument('--hop_length',type=int,default=128)
    parser.add_argument('--dt',type=float,default=4e-3)
    parser.add_argument('--min_max_norm',action="store_true")
    parser.add_argument('--do_norm',action="store_true")
    parser.add_argument('--lpff_size',default=128,type=int)
    parser.add_argument('--n_mels',default=80,type=int)

    args = parser.parse_args()

    print('Args:')
    print(args)
    if args.wth < 0:
        r = run_test_case(args)
    else :
        max_accs = []
        first_initialization = None
        wth_mask = None
        print('Training multiple time following the winning ticket')
        for winning_ticket_iter in range(args.resume_wth,args.wth):
            max_val_acc_i, first_initialization, wth_mask = run_test_case(args,winning_ticket_iter,first_initialization, wth_mask)
            print('In ticket iteration',winning_ticket_iter,'the best accuracy was',max_val_acc_i)
            max_accs.append(max_val_acc_i)
        print('Best Accuracies by iteration',max_accs)
    print('The End')
