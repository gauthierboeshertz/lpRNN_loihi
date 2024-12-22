import numpy as np
import os, sys
import argparse
import matplotlib.pyplot  as plt
import pickle
from utils import *
import json
np.random.seed(1)


def spikes_per_neuron(weights,input,input_idx):
    dir = 'plots/spikes/'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    snn = brianNetFromWeights(ts_ann=ts_ann,a=a,Iin=Iin,weights=weights)
    for layer in snn.layers:
        layer.probe(['spikes'])
    
    if args.brian:
        snn.refill_objects()
    snn.run(input)  

    spikes_per_neuron = []
    layer_names = []
    total_spikes = 0
    print('Sparsity') 
    spikes_per_layer_dict = {}
    for layer in snn.layers:
        layer_spikes = layer.spikes
        spikes_per_layer_dict[layer.name] = layer_spikes.sum()
        spikes_per_neuron.append(layer_spikes.sum()/layer.size)
        layer_names.append(layer.name)
        total_spikes += layer_spikes.sum()

        print(f'Layer {layer.name} spiked {layer_spikes.sum()} times for the input' )

        for spars_idx,spars in enumerate(args.sparsities):
            print('In layer '+layer.name,": ",((layer_spikes.sum(axis=1)/layer_spikes.shape[1])>spars).sum(), ' neurons fired more than ',spars*100,' % of the time')
       #     if spars_idx == 0:
       #         print(layer_spikes.sum(axis=1))
    print('The whole network spiked ',total_spikes)
    if args.save_plots:
        plt.bar(np.arange(0,len(layer_names)),np.array(spikes_per_neuron),width=0.4,tick_label=layer_names)
        plt.title('Spikes per neuron per layer on a random input')
        plt.ylabel('Spikes / neuron ')
        plt.xlabel('Layers')
        plt.savefig(dir+'spikes_per_neuron_per_layer'+str(args.wth)+'_'+str(input_idx)+'.jpg')
        plt.clf()        
        print('saved the plot in',dir+'spikes_per_neuron_per_layer'+str(args.wth)+'_'+str(input_idx)+'.jpg')

    return spikes_per_layer_dict


def weights_histo(weights):
    dir = 'plots/weights/'
    if not os.path.isdir(dir):
        os.mkdir(dir)
    for layer in weights['order']:
        ws = weights[layer]
        kernel = ws[0]
        print(layer)
        hist,edges =np.histogram(kernel, bins=np.unique(kernel).shape[0])
        plt.bar(np.unique(kernel).tolist(),hist,width=0.1,tick_label=np.unique(kernel).tolist())
        plt.title('Histograms of kernel of layer '+layer)
        plt.ylabel('Number of weights ')
        plt.xlabel('Weights value')
        plt.savefig(dir+'/'+layer+'_kernel.jpg')
        plt.clf()        
        if 'rnn' in layer:
            rec = ws[1]
            hist,edges =np.histogram(rec, bins=np.unique(rec).shape[0])
            plt.bar(np.unique(rec).tolist(),hist,width=0.1,tick_label=np.unique(rec).tolist())
            plt.title('Histograms of recurrent weights of layer '+layer)
            plt.ylabel('Weights value')
            plt.xlabel('Number of weights')
            plt.savefig(dir+'/'+layer+'_rec.jpg')
            plt.clf()        

def possible_params(weights):
    snn = brianNetFromWeights(ts_ann=ts_ann,a=a,Iin=Iin,weights=weights)
    mode = 'dense' if all([l.mode == 'dense' for l in snn.layers]) else 'lpRNN'
    lowest_wann = 100
    for w in weights.keys():
        if 'order' == w:
            continue
        lowest_wann = min(lowest_wann,float(np.min(np.abs(weights[w][0][weights[w][0]!=0]))))
        if len(weights[w]) == 3:
            lowest_wann = min(lowest_wann,float(np.min(np.abs(weights[w][1][weights[w][1]!=0])))) 
    
    plot_params(tau=args.tau,Iin=Iin,lowest_wann=lowest_wann,mode =mode,decay_bounds=(100,500),wfb_bounds=(100,1024) )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=str, default='',
                    help='What to plot, weights for weight histogram; spikes for spikes; energy for energy')
    parser.add_argument('--nunits', type=int, default=256,
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
    parser.add_argument('--wth', type=int, default=-1,
                        help='Winning ticket hypothesis iteration')
    parser.add_argument('--sparsities', type=str, default='',
                        help='To use when counting spike')
    parser.add_argument('--input_idx', type=int, default=-1,
                        help='Winning ticket hypothesis iteration')
    parser.add_argument('--last_ts', type=int, default=-100)
    parser.add_argument('--time_between_inputs', type=int, default=100,help="Measured in loihi timesteps")
    parser.add_argument('--brian', action='store_true')
    parser.add_argument('--num_inputs', type=int,default=-1)
    parser.add_argument('--save_plots', action='store_true')
    parser.add_argument('--activity_reg_weight', type=float, default=-1)
    parser.add_argument('--task_n_words',type=int,default=36,
                        help='gsc task')
    parser.add_argument('--augmented',action='store_true',
                        help='gsc task')
    parser.add_argument('--energy_accuracy',action='store_true',
                        help='Add probes to measure accuracy while doing energy measurements, just to debug, will slow down the simulation')
    parser.add_argument('--small_net',action='store_true')
    parser.add_argument('--ts_snn',type=float,default=-1,help="Time step of the snn, if not given, will be the lowest possible given the tau and ts_ann")
    parser.add_argument('--factor',type=float,default=-1,help="Factor to multiply the weights by in the SNN")

    args = parser.parse_args()
    print(args)

    nunits = str(args.nunits)
    ret_ratio = np.exp(-args.ts_ann/args.tau)
    
    if args.sparsities:
        args.sparsities = [float(spars) for spars in args.sparsities.split(',')]
    if args.wth >=0:
        model_save_path = 'data/'+ ('small_'if str(args.small_net) else '' )+'QlpRNN_'+str(args.nlayers)+'_'+str(args.nunits)+'_'+str(ret_ratio)+'_'+str(args.frac_bits)+'_nWTH_'+str(args.wth) +'_actreg'+str(args.activity_reg_weight)  +'_'+str(args.task_n_words)+'_'+str(args.augmented)+'.pth'
    else:
        model_save_path = 'data/'+ ('small_'if str(args.small_net) else '' )+'QlpRNN_'+str(args.nlayers)+'_'+str(args.nunits)+'_'+str(ret_ratio)+'_'+str(args.frac_bits)+'_actreg'+str(args.activity_reg_weight)  +'_'+str(args.task_n_words)+'.pth'


    with open(model_save_path, 'rb') as handle:
        weights = pickle.load(handle)
    print('Loaded weights from '+model_save_path)

    tstop = 1  # Simulation stop time in seconds
    ts_ann = 1 / 125  ##Timestep after the melspectogram
    num_steps = int(np.round(1 / ts_ann))  # np.arange(0, 1, ts_ann).shape[0]
    tau = args.tau  ### convert retention ratio to tau
    Iin = 6
    a = np.exp(-args.ts_ann/args.tau)

    if 'weights' in args.plot:
        weights_histo(weights)
    if 'possible_params' in args.plot:
        possible_params(weights)

    inputs =np.load('data/test_data_'+str(args.task_n_words)+ '.npy')


    if  args.input_idx > -1:
        input_idxs = np.array([args.input_idx])
    else :
        if args.num_inputs == -1:
            input_idxs = np.arange(len(inputs))
        else:
            input_idxs = np.random.randint(0,len(inputs),args.num_inputs)
    print(input_idxs)



    if 'spikes' in args.plot:
        total_spikes_per_layer_dict = {l: [] for l in weights['order']}
        for test_input_idx in input_idxs:
            input = inputs[test_input_idx]
            if not args.brian:
                input = input[0]
            if 'spikes' in args.plot:
                spikes_per_layer_dict_ = spikes_per_neuron(weights,input,test_input_idx)
                for l in spikes_per_layer_dict_ :
                    total_spikes_per_layer_dict[l].append(spikes_per_layer_dict_[l])

        for l in total_spikes_per_layer_dict:
            l_spikes = np.array(total_spikes_per_layer_dict[l])
            print('Layer', l, ' mean number of spikes', np.mean(l_spikes))
            print('Layer', l, ' std number of spikes', np.std(l_spikes))
            print('Layer', l, ' max number of spikes', np.max(l_spikes))
            print('Layer', l, ' min number of spikes', np.min(l_spikes))



