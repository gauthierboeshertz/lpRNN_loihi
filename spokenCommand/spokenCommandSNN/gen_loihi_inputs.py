# %%
import copy
import os
import numpy as np
import pickle
import numpy as np
from brian2.only import *
import argparse
np.random.seed(42)
import time
from utils import *
import pickle
from joblib import Parallel, delayed
import brian2

#brian2.clear_cache('cython')
prefs.codegen.target = 'numpy'
brian2.logging.file_log = False

def gen_inputs_for_loihi(input_idx):
    start_scope()
    if input_idx % 500 == 0:
        print('DOING TEST',input_idx)
    ip = inputs[input_idx]
    #
    probes = {str(0):['spikes'],str(-1):['i_fb','spikes','i_ip','i_in']}
    brian_net = brianNetFromWeights(ts_ann=ts_ann,a=a,Iin=Iin,weights=weights,probes=probes,ts_snn=ts_snn,factor=factor,optimal_snn=args.optimal_snn)
    brian_net.run(ip)        
    spikes = brian_net.layers[0].get_spikes_for_loihi()
    out_brian_fb = brian_net.layers[-1].i_fb
    out_brian_ip = brian_net.layers[-1].i_ip#np.argmax(np.mean(brian_net.layers[-1].i_ip[:,args.last_ts:],axis=1))
    out_brian_spikes = brian_net.layers[-1].spikes
    out_brian_in = brian_net.layers[-1].i_in

    pred_brian_fb = np.argmax(np.mean(out_brian_fb,axis=1))
    pred_brian_ip = np.argmax(np.mean(out_brian_ip,axis=1))
    pred_brian_spikes = np.argmax(out_brian_spikes.sum(axis=1))
    pred_brian_in = np.argmax(np.mean(out_brian_in,axis=1))

    ann = ANNFromWeights(ts_ann=ts_ann,a=a,Iin=Iin,weights=weights) 
    ann.run(ip)
    out_ann = np.argmax(np.mean(ann.layers[-1].state,axis=0))
    return spikes,pred_brian_fb,out_ann,labels[input_idx],pred_brian_spikes,pred_brian_ip,pred_brian_in,input_idx



parser = argparse.ArgumentParser()

parser.add_argument('--nunits', type=int, default=256,
                help='number of hidden layer units')
parser.add_argument('--last_ts', type=int, default=-100,
                help='number of hidden layer units')
parser.add_argument('--ts_ann', type=float, default=0.008, 
                help='number of hidden layer units')
parser.add_argument('--tau', type=float, default=0.035851360941796404,##tau for a retention ratio of 0.8
                    help='NUmber of integer weight bits')#               and timestep of 1/125
parser.add_argument('--nlayers', type=int, default=2,##tau for a retention ratio of 0.8
                    help='NUmber of integer weight bits')#               and timestep of 1/125
parser.add_argument('--int_bits', type=int, default=32,
                    help='NUmber of integer weight bits')
parser.add_argument('--frac_bits', type=int, default=3,
                    help='NUmber of fractional weight bits')
parser.add_argument('--num_ipts', type=int, default=-1,
                    help='NUmber of fractional weight bits')
parser.add_argument('--wth', type=int, default=-1)
parser.add_argument('--ret_ratio', type=float, default=0.8)
parser.add_argument('--nthreads', type=int, default=5)
parser.add_argument('--task_n_words',type=int,default=36,
                    help='gsc task')
parser.add_argument('--ts_snn',type=float,default=-1,#default=0.0005601775147155688,
                    help='gsc task')
parser.add_argument('--factor',type=float,default=-1,#default=349525.3333333333,
                    help='gsc task')
parser.add_argument('--optimal_snn',action='store_true')
parser.add_argument('--augment',action='store_true')
parser.add_argument('--lpff_size',type=int,default=128)

args =parser.parse_args()

print(args)

nunits = str(args.nunits)
model_save_path = './data/'+ 'QlpRNN_'+str(args.lpff_size)+"_"+str(args.nlayers)+'_'+str(args.nunits)+'_'+str(args.ret_ratio)+'_'+str(args.frac_bits)+'_nWTH_'+str(args.wth) +'_'+str(args.task_n_words)+("_augment" if args.augment else "")+'.pth'

with open(model_save_path, 'rb') as handle:
    weights = pickle.load(handle)
print('Loaded weights from '+model_save_path)
print('weights',weights['order'])

input_path = 'data/'+'test_data_'+str(args.task_n_words)+'.npy'
labels_path = 'data/'+'test_labels_'+str(args.task_n_words)+'.npy'

inputs = np.load(input_path)
print('Loaded inputs from ')
print('Input shape is ' ,inputs[0,:,:].shape)    
print(inputs.shape)

labels = np.load(labels_path)
print('Loaded labels ')
# Get inputs from bottom melspectrum 
tstop = 1  # Simulation stop time in seconds
ts_ann = 1/inputs.shape[-1]  ##Timestep of the melspectogram
a = args.ret_ratio  ##retention ratio
Iin = 6

if args.optimal_snn:
    print("Using ts {} and factor {} from arguments ".format(args.ts_snn,args.factor))
    ts_snn = args.ts_snn
    factor = args.factor
    
else:
    if args.ts_snn != -1:
        assert args.factor != -1
        ts_snn = args.ts_snn    
        factor = args.factor
    else:
        brian_net_ = brianNetFromWeights(ts_ann=ts_ann,a=a,Iin=Iin,weights=weights,optimal_snn=args.optimal_snn)
        ts_snn = brian_net_.layers[0].ts_snn
        factor = brian_net_.layers[0].factor
        print('ts snn',ts_snn)
        print('factor',factor)

num_ipts = inputs.shape[0] if args.num_ipts == -1 else args.num_ipts

#loihi_inputs = [gen_inputs_for_loihi)(test_num) for test_num in range(num_ipts)]
stime = time.time()
loihi_inputs = Parallel(n_jobs=args.nthreads)(delayed(gen_inputs_for_loihi)(int(test_num)) for test_num in range(num_ipts))
print('time taken',time.time()-stime)
ann_true = 0
brian_fb = 0
ann_brian = 0
brian_spikes = 0
brian_ip = 0
brian_in = 0

for i in range(len(loihi_inputs)):
    if loihi_inputs[i][1] == loihi_inputs[i][3]:
        brian_fb += 1
    if loihi_inputs[i][6] == loihi_inputs[i][2]:
        ann_brian += 1
    if loihi_inputs[i][4] == loihi_inputs[i][3]:
        brian_spikes += 1
    if loihi_inputs[i][5] == loihi_inputs[i][3]:
        brian_ip += 1
    if loihi_inputs[i][2] == loihi_inputs[i][3]:
        ann_true += 1
    if loihi_inputs[i][6] == loihi_inputs[i][3]:
        brian_in += 1

print('ann accuracy ',ann_true/len(loihi_inputs))
print('brian fb accuracy',brian_fb/len(loihi_inputs))
print('brian spike accuracy',brian_spikes/len(loihi_inputs))
print('brian ip accuracy',brian_ip/len(loihi_inputs))
print('brian in accuracy',brian_in/len(loihi_inputs))
print('ann brian',ann_brian/len(loihi_inputs))
print('Len of loihi inputs',len(loihi_inputs))

