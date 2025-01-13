
# coding: utf-8
# # Neuromorphic speech command recognition model
# We reuse an existing code sourced from  https://github.com/manuvn/SpeechCmdRecognition.git.
# The original code provided us with a good and readily usable infrastructure to plug in the proposed low pass RNN models.
import os
os.environ['PYTHONHASHSEED'] = "0"

# Following stderr write disables "Using tensorflow backend" messages
import warnings 
warnings.filterwarnings(action='once')
from layers.q_utils import quantize_weight,rsetattr,rgetattr
import os
import math
import argparse
import torch
import numpy as np
import models
import torch.nn as nn
import torch.nn.functional as F
import speech_datasetv2 
import speech_dataset
from layers import *
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(optimizer, epoch):
  epochs_drop = 5
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True
#torch.use_deterministic_algorithms(True)



def get_accuracy( output, target):
  output = torch.argmax(output, dim=-1)
  accuracy = torch.sum(output == target)
  accuracy = accuracy.to(float) # batch first
  return accuracy*100

def get_loss_acc( outputs, targets,criterion):
    loss =  criterion(outputs,targets)
    acc = get_accuracy(outputs,targets)
    return loss,acc.cpu().data.numpy()

def train_epoch(model,optimizer,criterion,train_dataloader,previous_mask = None):
    total_acc, total_loss = 0, 0
    model.train()
    
    for train_batch_idx, train_batch in enumerate((train_dataloader)):
        
        train_data = train_batch[0].to(DEVICE)
        train_labels = train_batch[1].to(DEVICE)

        optimizer.zero_grad()
        y = model(train_data)
        loss,acc = get_loss_acc(y,train_labels,criterion)

        loss.backward()

        if previous_mask is not None:
            for name, weight in model.named_parameters():
                if ('bias' not in name and 'ratio' not in name) :
                    weight.grad.data = torch.where(previous_mask[name].to(DEVICE),weight.grad.data,torch.zeros_like(weight.grad.data).to(DEVICE))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.cpu().data.numpy()
        total_acc += acc
    
    return model,total_loss/(len(train_dataloader)),total_acc/(len(train_dataloader.dataset))

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
    meanacc = val_acc/(len(loader.dataset))
    return meanloss,meanacc

def run_test_case(args,winning_ticket_iter=-1,first_initialization=None,wth_mask=None):

    set_seed(args.seed)

    ret_ratio= args.ret_ratio


    vers_str = 'v2' if args.v2 else 'v1'
    if winning_ticket_iter < 0:
        model_path = 'weights'+'/' +  ("lstm" if args.use_lstm else "QlpRNN")  + "_" +str(args.lpff_size)+'_' + str(args.nlayers)+ '_' + str(args.nunits) +'intbits_'+str(args.int_bits)+'frac_bits'+ str(args.frac_bits)+str(ret_ratio) +'_'+str(args.task_n_words)+'.pth'
    else:
        if args.use_lstm:
            model_path = 'weights'+'/lstm_' + str(args.nlayers)+ '_' + str(args.nunits) +'_nWTH_'+str(winning_ticket_iter) +"_"+vers_str+'_'+str(args.task_n_words)+'.pth'
        else:
            model_path = 'weights'+'/'+args.model +"_" +str(args.lpff_size)+ '_' + str(args.nlayers)+ '_' + str(args.nunits) +'intbits_'+str(args.int_bits)+'frac_bits'+ str(args.frac_bits)+str(ret_ratio) +'_nWTH_'+str(winning_ticket_iter) +'_'+str(args.task_n_words)+("_augment" if args.augment else "")+f"seed{args.seed}"+'.pth'

    # Speech Data Generator 
    print("model path",model_path)
    criterion = nn.CrossEntropyLoss() 
    n_classes = 35 if args.v2 and args.task_n_words == 35 else 12 if args.v2 else 36
    model = models.small_lprnn_net(n_classes,
                                ret_ratio=ret_ratio,
                                int_bits=args.int_bits,
                                frac_bits=args.frac_bits,
                                input_size=40,
                                lpff_size=args.lpff_size,
                                nunits=args.nunits,
                                nlayers=args.nlayers,
                                dev=DEVICE,
                                use_lstm=args.use_lstm,
                                use_bias=True)
    
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
    
    if args.v2 and args.task_n_words==35:
        train_loader, val_loader, test_loader = speech_datasetv2.get_dataloaders('data/',args=args)
    else:
        print(f"Using {args.task_n_words} words "+("v2" if args.v2 else "v1"))
        train_loader, val_loader, test_loader = speech_dataset.get_dataloaders('data/',args=args)
    print("TRAIN LOADER SIZE",len(train_loader))

    optimizer = torch.optim.AdamW(lr=args.lr,params = model.parameters(),weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=0.7,
            patience=1,
            min_lr=1e-6,
        )
    
    last_lr = scheduler.get_last_lr()
    max_val_acc = -1
    best_model = None
    for cur_epoch in range(args.nepochs):
        if (cur_epoch%args.print_freq==0):
            print('IN EPOCH :',cur_epoch)
        model, train_loss, train_acc = train_epoch(model,optimizer,criterion,train_loader,previous_mask)
        val_loss,val_acc = validate(model,val_loader,criterion)

        if np.isnan(val_loss).any():
            print('NaN loss exiting early!')
            break
        print(f"Epoch {cur_epoch}: Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}")
        if scheduler is not None:
            if last_lr != scheduler.get_last_lr():
                print('Learning rate changed from',last_lr,'to',scheduler.get_last_lr())
                last_lr = scheduler.get_last_lr()
        if val_acc> max_val_acc:
            #print('saving model to memory, current accuracy ',val_acc,'is better than previous',max_val_acc)
            max_val_acc = val_acc
            best_model = model.state_dict()
        
        scheduler.step(val_acc/100)

    print('FINISHED TRAINING')
    model.load_state_dict(best_model)
    if previous_mask is not None:
        new_mask = dict()
        for name, weight in model.named_parameters():
            if 'b' not in name and 'ratio' not in name:
                q_weight = quantize_weight(weight,args.int_bits,args.frac_bits)
                new_mask[name] = q_weight != 0
        count_pruned_weights(new_mask)

    torch.save(best_model,model_path)
    test_loss,test_acc = validate(model,test_loader,F.cross_entropy)
    
    print("TEST ACCURACY",test_acc)
    if winning_ticket_iter < 0:
        return max_val_acc  
    else:
        return max_val_acc, first_initialization,new_mask



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
    parser.add_argument('--frac_bits', type=int, default=3,
                        help='NUmber of fractional weight bits')
    parser.add_argument('--use_lstm', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--clipvalue', type=float, default=5,
                        help='Clip gradient value for optimizer. 0=disabled')
    parser.add_argument('--nepochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--wth', type=int, default=-1,
                        help='winning ticket hypothesis number of iterations')
    parser.add_argument('--resume_wth', type=int, default=0,
                        help='winning ticket hypothesis number of iterations')
    parser.add_argument('--augment',action='store_true',
                        help='winning ticket hypothesis number of iterations')
    parser.add_argument('--task_n_words',type=int,default=36,help='gsc task')
    parser.add_argument('--v2',action="store_true")
    parser.add_argument('--print_freq',type=int,default=10)
    parser.add_argument('--lpff_size',type=int,default=128)
    parser.add_argument('--ret_ratio',type=float,default=0.8)
    parser.add_argument('--weight_decay',type=float,default=1e-5)

    parser.add_argument('--seed',type=int,default=0)

    args = parser.parse_args()

    print('Args:')
    print(args)
    if args.wth < 0:
        run_test_case(args)
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
