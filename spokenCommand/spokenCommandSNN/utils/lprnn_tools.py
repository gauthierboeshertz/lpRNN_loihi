"""
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
from brian2.only import *
from scipy.signal import resample
from utils.loihi_equations import loihi_eq_dict, loihi_syn_dict, set_params, add_clipping_to_NeuronGroup
from tqdm import tqdm


def concat_loihi_inputs(input1,input2,num_inputs_before2,input_duration,time_between_inputs):
    nsteps_before_2 = num_inputs_before2*input_duration + num_inputs_before2*time_between_inputs
    new_input = [[] for _ in range(len(input1))]
    for i in range(len(input1)):
        spike_list1 = input1[i]
        new_input[i].extend(spike_list1)
        spike_list2 = input2[i]
        for spike2 in spike_list2:
            new_input[i].append(spike2+nsteps_before_2)
    return new_input


def load_weights(weight_file):
    with open(weight_file, 'rb') as handle:
        w = pickle.load(handle)
    # Put weights in format suitable for rnn.py
    weights = []
    Wbias = []
    weights += [w['qlp_rnn'][0].T]
    weights += [w['qlp_rnn'][1].T]
    Wbias += [w['qlp_rnn'][2].T]

    weights += [w['qlp_rnn_1'][0].T]
    weights += [w['qlp_rnn_1'][1].T]
    Wbias += [w['qlp_rnn_1'][2].T]

    weights += [w['q_dense'][0].T]
    Wbias += [w['q_dense'][1].T]

    weights += [w['q_dense_1'][0].T]
    Wbias += [w['q_dense_1'][1].T]

    weights += [w['q_dense_2'][0].T]
    Wbias += [w['q_dense_2'][1].T]

    for w in weights:
        print(w.shape)

    return weights, Wbias

def load_weights_new_net(weight_file):
    with open(weight_file, 'rb') as handle:
        w = pickle.load(handle)
    # Put weights in format suitable for rnn.py
    print(w.keys())
    
    weights = []
    Wbias = []
    weights += [w['qlp_rnn_1'][0].T]
    weights += [w['qlp_rnn_1'][1].T]
    Wbias += [w['qlp_rnn_1'][2].T]
    
    weights += [w['qlp_ff_1'][0].T]
    weights += [w['qlp_ff_1'][1].T]
    Wbias += [w['qlp_ff_1'][2].T]

    weights += [w['qlp_rnn_2'][0].T]
    weights += [w['qlp_rnn_2'][1].T]
    Wbias += [w['qlp_rnn_2'][2].T]

    weights += [w['q_dense_1'][0].T]
    Wbias += [w['q_dense_1'][1].T]

    weights += [w['q_dense_2'][0].T]
    Wbias += [w['q_dense_2'][1].T]

    weights += [w['q_dense_3'][0].T]
    Wbias += [w['q_dense_3'][1].T]

    for w in weights:
        print(w.shape)
    
    return weights, Wbias


def get_brian_spikes(use_saved_spikes,use_temp_file,inputs_idx,weights,bias,Iin,a,ts_ann,ts_snn,factor,inputs,time_between_ipts,data_dir="data"):
    
    l1Size=bias.shape[0]
    spikes_times_loihi = [[] for i in range(l1Size)]
    bias = [bias*factor/((-ts_ann/np.log(a))/ts_snn)]
    Iin = Iin*factor
    weights[0] = weights[0]*factor
    if use_saved_spikes:
        input_file = os.path.join(data_dir, 'all_spikes_ts_' + str(ts_snn) + '_ret_' + str(a) + '.pickle')
        if os.path.exists(input_file):
            print('brian input exists')
            with open(os.path.join(data_dir, input_file), 'rb') as handle:
                temp_spikes = pickle.load(handle)
            for i, rnd_idx in enumerate(inputs_idx):
                for j in range(len(temp_spikes[rnd_idx])):
                    spikes_times_loihi[j] += (np.array(temp_spikes[rnd_idx][j], dtype=int) + (i * int(1 / ts_snn)) + (
                        i) * time_between_ipts).tolist()
        else:
            raise Exception("No file with all the spikes wit hgiven tau and ts snn")

    elif use_temp_file:
        input_file = os.path.join(data_dir,  'temp_ts_' + str(ts_snn) + '_ret_' + str(a) + '_' + str(num_ipts) + '.pickle')
        if os.path.exists(input_file):
            print('brian input exists')
            with open(os.path.join(data_dir, input_file), 'rb') as handle:
                spikes_times_loihi = pickle.load(handle)
        else:
            print('brian input  doesnt exist')
            for i, idx in enumerate(tqdm(inputs_idx)):
                spike_times = get_spikes_reclayer_for_loihi(ip=inputs[idx].T, weights=weights,
                                                            bias=bias, Iin=Iin, a=a, ts_ann=ts_ann,
                                                            ts_snn=ts_snn)
                for j in range(len(spike_times)):
                    spikes_times_loihi[j] += ((np.round(spike_times[j] / (ts_snn * second) + 2) + (
                            i * int(1 / ts_snn)) + (i) * time_between_ipts).tolist())
            with open(os.path.join(data_dir, input_file), 'wb+') as handle:
                pickle.dump(spikes_times_loihi, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print("brian input doesn't exist")
        for i, idx in enumerate(tqdm(inputs_idx)):
            spike_times = get_spikes_reclayer_for_loihi(ip=inputs[idx].T, weights=weights,
                                                        bias=bias, Iin=Iin, a=a, ts_ann=ts_ann,
                                                        ts_snn=ts_snn)
            for j in range(len(spike_times)):
                spikes_times_loihi[j] += ((np.round(spike_times[j] / (ts_snn * second) + 2) + (i * int(1 / ts_snn)) + (
                    i) * time_between_ipts).tolist())
    
    return spikes_times_loihi


### change input to retention ratio per tau
# tau_from_ret_ratio
def create_taus(tauI=1, tauS=1, tauV=1, SIM_DT=10e-6):
    ####

    # tau = -ts_ann/np.log(a)

    tauV = (tauV / SIM_DT)
    tauI = (tauI / SIM_DT)
    tauS = (tauS / SIM_DT)

    return tauV, tauI, tauS


def get_spikes_reclayer_for_loihi(ip, weights, bias, Iin, a, ts_ann, ts_snn):
    start_scope()

    ip = resample(ip, int(1 / ts_snn), axis=1)

    l1Size = weights[0].shape[0]

    defaultclock.dt = ts_snn * second
    num_steps = ip.shape[1]

    simulation_time = num_steps * ts_snn * second

    for wb in bias:
        wb.shape = (wb.shape[0],)

    thr = 0
    tau = -ts_ann / np.log(a)
    tauS = tau / ts_snn
    tauI = tau / ts_snn
    tauV = tau / ts_snn

    neuron_params = {}
    neuron_params['tau_v'] = tauV
    neuron_params['tau_in'] = tauS
    neuron_params['tau_ip'] = tauI
    neuron_params['tau_fb'] = tauS
    neuron_params['gain'] = tauV
    neuron_params['vThMant'] = thr / 64
    neuron_params['ref_p'] = 0

    ip_curr = np.dot(weights[0], ip).T

    ip_to_res = TimedArray(np.ascontiguousarray(ip_curr), dt=ts_snn * second)

    Wfb1 = -Iin / (64 * tauS)
    lplayer1 = NeuronGroup(l1Size, **loihi_eq_dict)

    set_params(lplayer1, neuron_params)
    lplayer1.run_regularly(''' i_ip += ip_to_res(t,i)/tauS''', dt=ts_snn * second)

    lplayer1.bias = bias[0]
    feedback_lplayer1 = Synapses(lplayer1, lplayer1, **loihi_syn_dict)
    feedback_lplayer1.connect(condition='i==j')
    feedback_lplayer1.weight[:] = Wfb1
    feedback_lplayer1.w_factor[:] = 1

    lp1_lp1 = Synapses(lplayer1, lplayer1, model='weight:1', on_pre='''i_in_post +=   weight*64 ''')
    lp1_lp1.connect()
    lp1_lp1.weight = (-weights[1].T * Wfb1 / tauS).flatten()

    add_clipping_to_NeuronGroup(lplayer1)

    lplayer1_mon = StateMonitor(lplayer1, ['i_fb', 'i_in', 'i_ip'], record=[i for i in range(l1Size)])

    spike_monitor = SpikeMonitor(lplayer1)

    # net = Network(lp1_lp1, lplayer1 ,lplayer1_mon ,spike_monitor)
    run(simulation_time)

    spikes = copy.deepcopy(spike_monitor.spike_trains())
    return spikes

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))  
def ann(inSize=1,  
            l0Size=1, 
            l1Size=1, 
            l2Size=1,
            l3Size=1,
            l4Size=1,
            l5Size=1,
            l6Size=1,
            l7Size=1,
            outSize=1,
            test_ip=np.ndarray((1,1)),
            Iin =1,
            tau= 0.001477256629974861,
            Ts = 0.5e-6,
            W =[],
            Wbias = []
           ):
    
    """
    inSize = input dimension
    outSize = output dimension
    resSize = size of the reservoir
    """
    tsteps = np.arange(0,test_ip.shape[1],1)

    # Generate graph for visualization
    # plt.savefig('./img/ESNgraph.pdf')
    
    x0 = np.zeros((l0Size,))
    d1 = np.zeros((l1Size,))
    d2 = np.zeros((l2Size,))
    x1 = np.zeros((l3Size,))
    x2 = np.zeros((l4Size,))
    d3 = np.zeros((l5Size,))
    d4 = np.zeros((l6Size,))
    readout = np.zeros((outSize,))
    op = np.zeros((outSize,len(tsteps)))


    t_ip  = np.zeros((inSize,len(tsteps)))

    Wbias[0].shape =(l0Size,)
    Wbias[1].shape =(l1Size,)
    Wbias[2].shape =(l2Size,)
    Wbias[3].shape =(l3Size,)
    Wbias[4].shape =(l4Size,)
    Wbias[5].shape =(l5Size,)
    Wbias[6].shape =(l6Size,)
    Wbias[7].shape =(outSize,)


    # the retention factor inside reservoir = exp(-nsamples_in_tau/a)
    # chosen so that in time 10*tau, the value of exponent is almost 0
    a = (np.exp(-Ts/tau))#*0.6
    for t in range(len(test_ip[0,:])):
        
        u = test_ip[:,t] 
        t_ip[:,t] = u + 0
        
        # n_ip = np.maximum(n_ip,0) # Apply relu
        x0.shape = (l0Size,)
        n_ip = np.clip(np.dot( W[0], u)+Wbias[0] , 0, Iin)
        x0 = a*x0 + (1-a)*n_ip

        n_ip1 = np.clip(np.dot(W[1],x0)+Wbias[1], 0, Iin)
        d1.shape = (l1Size,)
        d1 = n_ip1

        n_ip2 = np.clip(np.dot(W[2],d1)+Wbias[2], 0, Iin)
        d2.shape = (l2Size,)
        d2 = n_ip2

        x1.shape = (l3Size,)
        n_ip3 = np.clip(np.dot( W[3], d2)+Wbias[3] +np.dot(W[4],x1), 0, Iin)
        x1 = a*x1 + (1-a)*n_ip3# filtering inside the neuron

        x2.shape = (l4Size,)
        n_ip4 = np.clip(np.dot(W[5],x1)+Wbias[4] +np.dot(W[6],x2), 0, Iin) 
        x2 = a*x2 + (1-a)*n_ip4
        

        n_ip5 = np.clip(np.dot(W[7],x2)+Wbias[5], 0, Iin)
        d3.shape = (l5Size,)
        d3 = n_ip5
    

        n_ip6 = np.clip(np.dot(W[8],d3)+Wbias[6], 0, Iin)
        d4.shape = (l6Size,)
        d4 = n_ip6

        n_ip7 = softmax((np.dot(W[9],d4)+Wbias[7]))
        readout.shape = (outSize,)
        readout = n_ip7

    
        op[:,t] = readout
        
        
    return op 

def new_ann(inSize=1, l1Size=1, l2Size=1, l3Size=1, l4Size=1,l5Size=1, outSize=1,
        test_ip=np.ndarray((1, 1)),
        Iin=1, tau=0.001477256629974861, Ts=0.5e-6, W=[], Wbias=[]):
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    num_steps = test_ip.shape[1]

    # Generate graph for visualization
    # plt.savefig('./img/ESNgraph.pdf')

    x1 = np.zeros((l1Size,))
    x2 = np.zeros((l2Size,))
    x3 = np.zeros((l3Size,))

    d1 = np.zeros((l4Size,))
    d2 = np.zeros((l5Size,))
    readout = np.zeros((outSize,))
    op = np.zeros((outSize, num_steps))
    l1_states = np.zeros((l1Size, num_steps))
    l2_states = np.zeros((l2Size, num_steps))
    l3_states = np.zeros((l3Size, num_steps))
    l4_states = np.zeros((l4Size, num_steps))

    t_ip = np.zeros((inSize, num_steps))

    Wbias[0].shape = (l1Size,)
    Wbias[1].shape = (l2Size,)
    Wbias[2].shape = (l3Size,)
    Wbias[3].shape = (l4Size,)
    Wbias[4].shape = (l5Size,)
    Wbias[5].shape = (outSize,)

    # the retention factor inside reservoir = exp(-nsamples_in_tau/a)
    # chosen so that in time 10*tau, the value of exponent is almost 0
    a = (np.exp(-Ts / tau))  # *0.6
    max_act = [np.zeros((l1Size,)), np.zeros((l2Size,)), np.zeros((l3Size,)), np.zeros((l4Size,))]
    for t in range(len(test_ip[0, :])):
        u = test_ip[:, t]
        t_ip[:, t] = u + 0

        # n_ip = np.maximum(n_ip,0) # Apply relu
        x1.shape = (l1Size,)
        n_ip = np.clip(np.dot(W[0], u) + Wbias[0] + np.dot(W[1], x1), 0, Iin)
        x1 = a * x1 + (1 - a) * n_ip  # filtering inside the neuron

        x2.shape = (l2Size,)
        n_ip2 = np.clip(np.dot(W[2], x1) + Wbias[1] , 0, Iin)
        x2 = a * x2 + (1 - a) * n_ip2
        
        
        n_ip3 = np.clip(np.dot(W[4], x2) + Wbias[2] + np.dot(W[5], x3), 0, Iin)
        x3.shape = (l3Size,)
        x3 = a * x3 + (1 - a) * n_ip3


        n_ip4 = np.clip(np.dot(W[6], x3) + Wbias[3], 0, Iin)
        d1.shape = (l4Size,)
        d1 = n_ip4


        n_ip5 = np.clip(np.dot(W[7], d1) + Wbias[4], 0, Iin)

        d2.shape = (l5Size,)
        d2 = n_ip5

        n_ip6 = softmax(np.dot(W[8], d2) + Wbias[5])
        readout.shape = (outSize,)
        readout = n_ip6

        # Save state transient output

        readout.shape = (outSize,)
        op[:, t] = readout

    return  op


def new_ann_last_layers(inSize=1, l1Size=1, l2Size=1, l3Size=1, l4Size=1,l5Size=1, outSize=1, test_ip=np.ndarray((1, 1)),
        Iin=5,  W=[], Wbias=[]):
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    num_steps = test_ip.shape[1]

    # Generate graph for visualization
    # plt.savefig('./img/ESNgraph.pdf')

    x1 = np.zeros((l1Size,))
    x2 = np.zeros((l2Size,))
    x3 = np.zeros((l3Size,))

    d1 = np.zeros((l4Size,))
    d2 = np.zeros((l5Size,))
    readout = np.zeros((outSize,))
    op = np.zeros((outSize, num_steps))
    l1_states = np.zeros((l1Size, num_steps))
    l2_states = np.zeros((l2Size, num_steps))
    l3_states = np.zeros((l3Size, num_steps))
    l4_states = np.zeros((l4Size, num_steps))

    t_ip = np.zeros((inSize, num_steps))

    Wbias[0].shape = (l1Size,)
    Wbias[1].shape = (l2Size,)
    Wbias[2].shape = (l3Size,)
    Wbias[3].shape = (l4Size,)
    Wbias[4].shape = (l5Size,)
    Wbias[5].shape = (outSize,)

    # the retention factor inside reservoir = exp(-nsamples_in_tau/a)
    # chosen so that in time 10*tau, the value of exponent is almost 0
    for t in range(len(test_ip[0, :])):
        u = test_ip[:, t]

        n_ip4 = np.clip(np.dot(W[6], u) + Wbias[3], 0, Iin)
        d1.shape = (l4Size,)
        d1 = n_ip4


        n_ip5 = np.clip(np.dot(W[7], d1) + Wbias[4], 0, Iin)

        d2.shape = (l5Size,)
        d2 = n_ip5

        n_ip6 = softmax(np.dot(W[8], d2) + Wbias[5])
        readout.shape = (outSize,)
        readout = n_ip6

        # Save state transient output

        readout.shape = (outSize,)
        op[:, t] = readout
    return  op

