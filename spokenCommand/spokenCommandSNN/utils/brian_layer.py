import numpy as np
from .loihi_equations import loihi_eq_dict, loihi_syn_dict, set_params,add_clipping_to_NeuronGroup,add_rounding_to_NeuronGroup,add_relu_to_NeuronGroup
from math import isclose
import copy
from brian2.only import *


def calculate_effective_weight(numWeightBits=8, IS_MIXED=0, weight=255, weightExponent=0):
    '''
    calculates and prints the actual weight of a synapse after discretization
    Please compare with the following documentation file: /docs/connection.html
    :param numWeightBits: number of  weight bits
    :param IS_MIXED: is the sign mode mixed (1)? Set to 0 if sign mode is exc or inh.
    :param weight: the weight
    :param weightExponent: weight exponent
    :return: the effective weight
    '''
    numLsbBits = 8 - numWeightBits - IS_MIXED
    try:
        actWeight = (weight >> numLsbBits) << numLsbBits
    except  TypeError:

        actWeight = np.left_shift(np.right_shift(np.asarray(weight, dtype=int), numLsbBits), numLsbBits)

    # print('original weight:', weight)
    # print('actual weight:', actWeight)
    # print('num lsb:', numLsbBits)
    # print('weight (with exponent):', actWeight * 2 ** weightExponent)
    # print('weight effect on current (with exponent):', actWeight * 2 ** (6 + weightExponent))
    return actWeight


def calculate_mant_exp(value, precision=8, name=""):
    """
    This function calculates the exponent and mantissa from a desired value. Can e.g. be used for weights.
    If used for weights: Please use calculate_effective_weight to calculate
     the effective weight also taking into account the precision.

    Important: This is based on a very simple idea to maximize the precision.
     However, it does not replace manual checking of your weights
     (E.g. instead of letting them go from 0-300, you should restrict them to 0-255,
     as you will loose precision otherwise)

    Also, be careful when using this with plastic weights.
    Otherwise your range might be limited to the initial weight range.

    :param value: the value for which you want to calculate mantissa and exponent
    :param precision: the allowed precision in bits for the mantissa
    :return: mantissa, exponent
    """
    value = np.asarray(value)
    # print('desired value:', value)
    exponent = 0
    while np.abs(np.max(value)) >= (2 ** precision) and not exponent >= 8:
        value = value / 2
        exponent += 1

    while np.abs(np.max(value)) < (2 ** precision / 2) and np.abs(np.max(value)) != 0 and not exponent <= -8:
        value = value * 2
        exponent += -1

    value = np.asarray(np.round(value), dtype=int)
    # print('actual value of', name, ':', value * 2 ** exponent, 'mantissa:', value, 'exponent:', exponent)
    return value, exponent

class BrianLayer:
    def __init__(self, mode, size, ts_ann, Iin,
                 weights, bias, ts_snn=None, tau=None, ret_ratio=None, factor=None,ret_ratio_prev=None,in_lprnn_net=False,name='',optimal_snn=False):

        if mode == 'dense' and len(weights) == 2:
            raise ValueError(' Only input weights for a dense layer')
        
        self.mode = mode
        self.size = size
        self.ts_ann = ts_ann
        self.bias = copy.deepcopy(bias)
        self.weights = copy.deepcopy(weights)
        self.Iin = Iin
        self.in_lprnn_net = in_lprnn_net
        self.name = name
        self.optimal_snn = optimal_snn
        if ret_ratio != None:
            self.tau = -ts_ann / np.log(ret_ratio)
            self.tau_prev = -ts_ann / np.log(ret_ratio_prev)
        elif tau != None:
            self.tau = tau
            self.tau_prev = tau
        else:
            raise ValueError(' No tau or retention ratio given')
        self.ret_ratio = np.exp(-ts_ann/self.tau)
        self.ts_snn = ts_snn
        self.factor =factor

        if (self.factor == None or self.ts_snn == None):
            print("No factor or ts_snn given")
            self.ts_snn, self.factor = self.find_params()
        self.Iin_snn = self.Iin*self.factor
        self.tauU = self.tau_prev / self.ts_snn
        self.tauV = self.tau / self.ts_snn
        self.tauS = self.tau / self.ts_snn
        self.monitor = None
        self.brian2_objects = []
        self.input_length = None

        if mode == 'dense':
            self.tauI = 1
        else:
            self.tauI = self.tau / self.ts_snn

        self.fb_weight = - self.Iin_snn / (64 * self.tauS)
        

        self.snn_weights = []
        self.snn_weights += [self.weights[0] *  self.Iin_snn / (64 * self.tauI * self.tauU)]
        if mode == 'lpRNN':
            self.snn_weights += [self.weights[1] *  self.Iin_snn / (64 * self.tauI * self.tauU)]
            
        self.bias = self.bias*self.factor / (self.tauI*self.tauU)
        self.loihi_weights = []
        if not self.optimal_snn:
            for w in self.snn_weights:
                mant, exp = calculate_mant_exp(w)
                mant = calculate_effective_weight(weight=mant)
                self.loihi_weights.append(mant * 2 ** exp)
        else:
            for w in self.snn_weights:
                self.loihi_weights.append(w)


        self.neurons =  NeuronGroup(self.size, **loihi_eq_dict)
        neuron_params = {}
        neuron_params['tau_v'] = self.tauV
        neuron_params['tau_in'] = self.tauU
        neuron_params['tau_ip'] = self.tauI
        neuron_params['tau_fb'] = self.tauS
        neuron_params['gain'] = self.tauV
        neuron_params['vThMant'] = 0
        neuron_params['ref_p'] = 0

        set_params(self.neurons, neuron_params)
        self.neurons.bias = self.bias
        self.brian2_objects.append(self.neurons)

        self.connect_feedback()
        if mode == 'lpRNN':
            self.connect_recurrent_weights()

        if not self.optimal_snn:
            clip_group = add_clipping_to_NeuronGroup(self.neurons)
            self.brian2_objects.extend(clip_group)
            round_group = add_rounding_to_NeuronGroup(self.neurons)
            self.brian2_objects.extend(round_group)
        else:
            clip_group = add_relu_to_NeuronGroup(self.neurons)
            self.brian2_objects.extend(clip_group)


    def connect_recurrent_weights(self):
        self.recurrent_connection = Synapses(self.neurons, self.neurons, model ='weight:1', on_pre= '''i_in_post +=   weight*64 ''')
        self.recurrent_connection.connect()
        self.recurrent_connection.weight = self.loihi_weights[1].T.flatten()
        self.brian2_objects.append(self.recurrent_connection)

    def connect_from_spikegenerator(self, spikegen):
        for i  in range(spikegen.size):
            self.brian2_objects.append(spikegen.gen(i))
            self.input_connection = Synapses(spikegen.gen(i), self.neurons, model ='weight:1', on_pre= '''i_in_post +=   weight*64 ''')
            self.input_connection.connect()
            self.input_connection.weight = self.loihi_weights[0].T[i].flatten()
            self.brian2_objects.append(self.input_connection)

    def connect_from_layer(self, previous_layer):
        self.input_connection = Synapses(previous_layer.neurons, self.neurons, model ='weight:1', on_pre= '''i_in_post +=   weight*64 ''')
        self.input_connection.connect()
        self.input_connection.weight = self.loihi_weights[0].T.flatten()
        self.brian2_objects.append(self.input_connection)


    @property
    def i_fb(self):
        if self.monitor is None:
            raise Exception('No monitor')
        return -self.monitor.i_fb

    @property
    def i_ip(self):
        if self.monitor is None:
            raise Exception('No monitor')
        return self.monitor.i_ip

    @property
    def i_in(self):
        if self.monitor is None:
            raise Exception('No monitor')
        return self.monitor.i_in

    @property
    def spikes(self):
        if self.input_length is None:
            raise Exception('Input length is none')

        spike_times = self.get_spike_trains()
        spikes = np.zeros((self.size,self.input_length))
        loihi_spikes = [[] for i in range(self.size)]

        for j in range(len(spike_times)):
            loihi_spikes[j] += ((np.round(spike_times[j] / (self.ts_snn * second))).astype(np.int32).tolist())
        for i in range(self.size):
            try:
                spikes[i,loihi_spikes[i]] = 1
            except Exception as inst:
                print(loihi_spikes[i])
                print(spike_times[i])
                print(inst)
                exit()
        return spikes


    def probe(self, probes = None):
        '''
        probes = list of probes to use
        '''
        in_probes = [probe for probe in probes if probe in ['i_ip','i_in','i_fb']]
        if len(in_probes):
            self.monitor = StateMonitor(self.neurons, in_probes, record=[ i for i in range(self.size)])
            self.brian2_objects.append(self.monitor)
        
        if 'spikes' in probes:
            self.spike_monitor = SpikeMonitor(self.neurons)
            self.brian2_objects.append(self.spike_monitor)

    def get_spike_trains(self):
        return self.spike_monitor.spike_trains()

    def get_spikes_for_loihi(self):
        loihi_spikes = [[] for i in range(self.size)]
        ts_snn = self.ts_snn
        spike_times = self.get_spike_trains()
        for j in range(len(spike_times)):
            loihi_spikes[j] += ((np.round(spike_times[j] / (ts_snn * second) + 2)).tolist())

        return loihi_spikes

    def reset(self):
        if self.monitor is not None:
            self.brian2_objects.remove(self.monitor)
        self.monitor = None
        self.input_length = None
        

    def compute_bias_mant_exp(self, bias):
        exp = 0
        mant = bias
        while np.abs(mant) > 4095:
            exp += 1
            mant /= 2

        assert mant * 2 ** exp == bias
        return mant, exp

    
    def connect_feedback(self):
        self.feedback_connection =  Synapses(self.neurons, self.neurons, **loihi_syn_dict)
        self.feedback_connection.connect(condition='i==j')
        self.feedback_connection.weight[:] = self.fb_weight
        self.feedback_connection.w_factor[:] = 1
        self.brian2_objects.append(self.feedback_connection)
      
    def find_params(self):
        def get_mantexp(val):
            exp =0
            mant = copy.deepcopy(val)
            while np.abs(mant)> 255:
                mant /=2
                exp +=1

            if exp>=8:
                raise Exception(str(val)+' is too big to big for weights')

            return mant, exp
    
        def check_mantexp(val):
            mant,exp = get_mantexp(val)
            return mant == int(mant)

        lowest_wann = np.min(np.abs(self.weights[0][self.weights[0]!=0]))
        if self.mode == 'lpRNN' or self.in_lprnn_net:
            if self.ts_snn is not None:
                taus = self.tau_prev/self.ts_snn
                for wfb in range(100,4096):
                    f =  wfb*taus*64/(self.Iin)
                    if (f >= 2**23/self.Iin or f<1) and not self.optimal_snn:
                        continue
                    wsn = wfb*lowest_wann/taus 

                    if (isclose(float(wsn),float(np.round(wsn)),abs_tol=10**-3)  and wsn>=1)  :
                        return self.ts_snn,f 
                print('difint fini')
            for decay in range(1, 100):
                ts = decay*self.tau_prev/4096
                taus = self.tau_prev/ts
                for wfb in range(100,4096):
                    f =  wfb*taus*64/(self.Iin)
                    if (f > 2**23/(2*self.Iin)) and not self.optimal_snn:
                        continue
                        
                    wsn = wfb*lowest_wann/taus 
                    if (isclose(float(wsn),float(np.round(wsn)),abs_tol=10**-5)   and wsn>=1 ) or self.optimal_snn:
                        return ts,f

                    if not check_mantexp(wfb):
                        mant_,exp_ = get_mantexp(wfb)
                        Wfb_r = int(mant_)*2**exp_
                        f_r = Wfb_r *taus*64/self.Iin
                        if f_r > 2**23/self.Iin:
                            continue

                        wsn_r = Wfb_r*lowest_wann/taus 
                        if isclose(float(wsn_r),float(np.round(wsn_r)),abs_tol=10**-5)  and wsn_r>=1:
                            return ts,f_r
                        
        else :
            if self.ts_snn is not None:
                taus = self.tau_prev/self.ts_snn
                for wfb in range(200,4096):
                    f =  wfb*taus*64/(self.Iin)
                    if (f >= 2**23/self.Iin or f<1) and not self.optimal_snn:
                        continue
                    wsn = wfb*lowest_wann

                    if isclose(float(wsn),float(np.round(wsn)),abs_tol=10**-3)  and wsn>=1 :
                        return self.ts_snn,f                
            else:    
                for decay in range(5,100):
                    ts = decay*self.tau_prev/4096
                    taus = self.tau_prev/ts

                    for wfb in range(100,4048):
                        f =  wfb*taus*64/(self.Iin)
                        if f >= 2**23/self.Iin:
                            continue

                        wsn = wfb*lowest_wann
                        if isclose(float(wsn),float(np.round(wsn)),abs_tol=10**-3)  and wsn>=1 :
                            return ts,f

                        if not check_mantexp(wfb):
                            mant_,exp_ = get_mantexp(wfb)
                            Wfb_r = int(mant_)*2**exp_
                            f_r = Wfb_r *taus*64/self.Iin
                            if f_r >=2**23/self.Iin:
                                continue

                            wsn_r = Wfb_r*lowest_wann
                            if isclose(float(wsn_r),float(np.round(wsn_r)),abs_tol=10**-3)  and wsn_r>=1 :
                                return ts,f_r
        raise Exception('no params found!')


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        r_string = 'Timestep snn:' +str(self.ts_snn)+ '\n'
        r_string += 'factor:' +str(self.factor)+ '\n'
        r_string += 'not rounded decay:' + str((1 / self.tauV) * 2 ** 12) + '\n'
        r_string += 'tauV=' + str(self.tauV) + '\n'
        r_string += 'tauI=' + str(self.tauI) + '\n'
        r_string += 'tauS=' + str(self.tauS) + '\n'
        r_string += 'tauU=' + str(self.tauS) + '\n'
        r_string += 'feedback weight=' + str(self.fb_weight) + '\n'
        r_string += 'feedforward weights' + str(self.loihi_weights[0]) + '\n'
        if self.mode == 'lpRNN':
            r_string += 'recurrent weights' + str(self.loihi_weights[1]) + '\n'

        r_string += '  '
        r_string += 'difference between real weights and loihi weights :' + '\n'
        r_string += '     feedforward weights diff:' + str(np.sum(self.snn_weights[0] - self.loihi_weights[0])) + '\n'
        if self.mode == 'lpRNN':
            r_string += '    recurrent weights diff ' + str(np.sum(self.snn_weights[1] - self.loihi_weights[1])) + '\n'
        return r_string


