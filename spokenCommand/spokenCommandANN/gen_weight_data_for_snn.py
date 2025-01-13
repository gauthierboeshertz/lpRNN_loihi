
# coding: utf-8
# # Neuromorphic speech command recognition model
# We reuse an existing code sourced from  https://github.com/manuvn/SpeechCmdRecognition.git.
# The original code provided us with a good and readily usable infrastructure to plug in the proposed low pass RNN models.

# Following stderr write disables "Using tensorflow backend" messages
import pickle
import argparse
import torch
#import speech_dataset
import numpy as np
import models
import torch.nn as nn
from layers import quantize_weight,jit_lpRNN,jit_lpLinear,Dense
import speech_datasetv2 
import speech_dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        model_path = 'weights'+'/' +  ("lstm" if args.use_lstm else "QlpRNN")  + "_" +str(args.lpff_size)+'_' + str(args.nlayers)+ '_' + str(args.nunits) +'intbits_'+str(args.int_bits)+'frac_bits'+ str(args.frac_bits)+str(args.ret_ratio) +'_'+str(args.task_n_words)+'.pth'
    else:
        if args.use_lstm:
            model_path = 'weights'+'/lstm_' + str(args.nlayers)+ '_' + str(args.nunits) +'_nWTH_'+str(args.wth) +"_"+vers_str+'_'+str(args.task_n_words)+'.pth'
        else:
            model_path = 'weights'+'/'+args.model +"_" +str(args.lpff_size)+ '_' + str(args.nlayers)+ '_' + str(args.nunits) +'intbits_'+str(args.int_bits)+'frac_bits'+ str(args.frac_bits)+str(args.ret_ratio) +'_nWTH_'+str(args.wth) +'_'+str(args.task_n_words)+("_augment" if args.augment else "")+f"seed{args.seed}"+'.pth'

    print("MODEL PATH",model_path)
    
    if args.v2 and args.task_n_words==35:
        _, _, test_loader = speech_datasetv2.get_dataloaders('data/',args=args)
    else:
        print(f"Using {args.task_n_words} words "+("v2" if args.v2 else "v1"))
        _, _, test_loader = speech_dataset.get_dataloaders('data/',args=args)
        
    print(model_path)
    sr = 16000
    n_classes = 35 if args.v2 and args.task_n_words == 35 else 12 if args.v2 else 36
    model = models.small_lprnn_net(n_classes,
                                ret_ratio=args.ret_ratio,
                                int_bits=args.int_bits,
                                frac_bits=args.frac_bits,
                                input_size=40,
                                lpff_size=args.lpff_size,
                                nunits=args.nunits,
                                nlayers=args.nlayers,
                                dev=DEVICE,
                                use_lstm=args.use_lstm,
                                use_bias=True)

    model.load_state_dict(torch.load(model_path))


    model = model.to(args.device)
    model.eval()


    data_path = '../spokenCommandSNN/data/test_data'+'_'+str(args.task_n_words)+'.npy'
    label_path = '../spokenCommandSNN/data/test_labels'+'_'+str(args.task_n_words)+'.npy'
    model_save_path = '../spokenCommandSNN/data/'+ 'QlpRNN_'+str(args.lpff_size)+"_"+str(args.nlayers)+'_'+str(args.nunits)+'_'+str(args.ret_ratio)+'_'+str(args.frac_bits)+'_nWTH_'+str(args.wth) +'_'+str(args.task_n_words)+("_augment" if args.augment else "")+'.pth'
    save_weights(model,model_save_path)
    test_and_save_data(model,test_loader,nn.CrossEntropyLoss(),data_path,label_path)
    return 

def save_weights(model,model_path):
    q_model_dict = dict()
    q_model_dict['order'] = []
    for l_idx,layer in (enumerate(model.modules())):
        if isinstance(layer, jit_lpRNN):
            w = [quantize_weight(layer.linear.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),quantize_weight(layer.linear_hh.weight.data,args.int_bits,args.frac_bits).cpu().numpy(),quantize_weight(layer.linear.bias.data,args.int_bits,args.frac_bits).cpu().numpy()]
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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--wth', type=int, default=-1,
                        help='winning ticket hypothesis iteration')
    parser.add_argument('--task_n_words',type=int,default=36,
                        help='gsc task')
    parser.add_argument('--v2',action="store_true")
    parser.add_argument('--lpff_size',type=int,default=128)
    parser.add_argument('--ret_ratio',type=float,default=0.8)
    parser.add_argument('--seed',type=int,default=0)

    args = parser.parse_args()
    print(args)
    save_data(args)

