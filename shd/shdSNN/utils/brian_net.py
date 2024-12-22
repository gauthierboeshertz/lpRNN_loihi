from typing import List
import numpy as np
from .brian_layer import BrianLayer
from brian2.only import *
from brian2.core.network import Network
from scipy.signal import resample
import copy
import pickle
import math

def brianNetFromWeights(ts_ann,a,Iin,path=None,weights=None,ts_snn=None,optimal_snn=False,factor=None,probes = None):
  """
  The path should be a pickle file which is a dictionnary with the name of the layer mapping to it's weights
    There should also be an 'order' key which maps to the order of the layers using the names
  """
  Iin = copy.deepcopy(Iin)
  factor = copy.deepcopy(factor)
  if path is not None:
    with open(path, 'rb') as handle:
      weights = pickle.load(handle)

  order = weights['order']

  layers = []
  in_lprnn_net = any(['lp' in layer for layer in order])
  for layer in order:
    is_lprnn_layer = 'lp' in layer
    w_b = weights[layer]
    w = [w_b[i] for i in range(1+int(is_lprnn_layer))]
    if 'ff' in layer and is_lprnn_layer :
      w[1] = np.zeros((w[0].shape[0],w[0].shape[0]))
    bias = w_b[-1].T
          
    if not optimal_snn:
      if ts_snn == 0.0005601775147155688 or ts_snn is None:
        if is_lprnn_layer:
            factor = 349525.3333333333
        if not is_lprnn_layer :
            factor = 131072.0
        if layer == order[-1]:
            factor = 32768.0
      else:
        pass
        # assert ts_snn == 0.003361065088293413
        # if layer == order[-1]:
        #     factor = 4551.11111111111

      

    layers.append(BrianLayer( size=w[0].shape[0], mode='lpRNN' if is_lprnn_layer else 'dense' ,
                        ts_ann=ts_ann,
                        ret_ratio=a,
                        ret_ratio_prev=a,
                        Iin=Iin,
                        weights=w,
                        bias=bias,
                        ts_snn=ts_snn,
                        optimal_snn=optimal_snn,
                        factor=factor,
                      in_lprnn_net=in_lprnn_net,name=layer))
  if probes is not None:
    for layer_to_probe in probes:
      layers[int(layer_to_probe)].probe(probes[layer_to_probe])
  briannet = BrianNet(layers)
  return briannet



class BrianNet:
  def __init__(self,layers : List[BrianLayer]):
      self.layers = layers
      self.net = Network()

      for layer_idx in range(1,len(layers)):
        self.layers[layer_idx].connect_from_layer(self.layers[layer_idx-1])
      
      for layer in layers:
        self.net.add(*layer.brian2_objects)
  
  def refill_objects(self):
    self.net = Network()
    for layer in self.layers:
      self.net.add(*layer.brian2_objects)

  def reset(self):
    for layer in self.layers:
      layer.reset()

  def add_preprocessing_layer(self):
    
    size = self.layers[0].snn_weights[0].shape[1]


    prepro_layer =BrianLayer( size=size, mode= 'dense' ,
                        ts_ann= self.layers[0].ts_ann,
                        ret_ratio= self.layers[0].ret_ratio,
                        ret_ratio_prev=self.layers[0].ret_ratio,
                        Iin=self.layers[0].Iin,
                        weights=[np.eye(size)],
                        bias=np.zeros((size,)),
                        ts_snn=self.layers[0].ts_snn,
                        optimal_snn=self.layers[0].optimal_snn,
                        factor=131072.0,
                      in_lprnn_net=self.layers[0].in_lprnn_net,name='preprocess_layer')
    self.layers.insert(0,prepro_layer)
    self.layers[1].connect_from_layer(self.layers[0])
    self.reset()
    self.refill_objects()


  def __repr__(self):
    r_string = 'Brian network \n'
    r_string +=  'Layers: '+ str(self.layers) +' \n'
    for l_idx,l in enumerate(self.layers):
      r_string += '--'*10 + 'Layer '+str(l_idx)+ '--'*10 
      r_string += str(l)
    return r_string
  def __str__(self):
      return self.__repr__()

  def run(self,inputs=None):
    """
    The input should be with shape : ( n features, n timestep)
    """
    inputs = copy.deepcopy(inputs)
    tauInp = self.layers[0].tauU
    ts_snn = self.layers[0].ts_snn
    ts_ann = self.layers[0].ts_ann
    defaultclock.dt = ts_snn*second
    nsteps = math.floor(inputs.shape[1]*(ts_ann/ts_snn))
    resampled_inputs = resample(inputs,nsteps,axis=1)
    resampled_inputs = np.dot(self.layers[0].weights[0],resampled_inputs)
    resampled_inputs = resampled_inputs * self.layers[0].factor
    brian_input = TimedArray(np.ascontiguousarray(resampled_inputs.T),dt =ts_snn*second)
    run_reg = self.layers[0].neurons.run_regularly(''' i_ip += brian_input(t,i)/tauInp ''', dt=ts_snn*second) 
    self.net.add(run_reg)
    self.net.run((nsteps*ts_snn*second))
    for layer in self.layers:
      layer.input_length = nsteps

