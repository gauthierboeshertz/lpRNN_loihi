import copy
from typing import List
import numpy as np
from .ann_layer import *
import copy
import pickle

def ANNFromWeights(ts_ann,a,Iin,path=None,weights=None):
  """
  The path should be a pickle file which is a dictionnary with the name of the layer mapping to it's weights
    There should also be an 'order' key which maps to the order of the layers using the names
  """
  Iin = copy.deepcopy(Iin)
  if path is not None:
    with open(path, 'rb') as handle:
      weights = pickle.load(handle)

  order = weights['order']

  layers = []
  for layer in order:
    w_b = weights[layer]
    is_lprnn_layer = 'lp' in layer
    w = [w_b[i] for i in range(1+int(is_lprnn_layer))]
    if 'ff' in layer and is_lprnn_layer :
      w[1] = np.zeros((w[0].shape[0],w[0].shape[0]))

    bias = w_b[-1].T
    layers.append(AnnLayer(size=w[0].shape[0], mode= 'lpRNN' if is_lprnn_layer else 'dense',
                        ts_ann=ts_ann,
                        ret_ratio=a,
                        Iin=Iin,
                        weights=w,
                        bias=bias,
                        clip_relu= layer != order[-1]
                    ))        
  ann = ANN(layers)
  return ann

class ANN:
  def __init__(self,layers : List[AnnLayer]):
      self.layers = layers
  
  def reset(self):
    for layer in self.layers:
      layer.reset()
  
  def run(self,inputs):
    """
    The input should be with shape : ( n features, n timestep)
    """
    inputs = copy.deepcopy(inputs)
    for layer_idx,layer in enumerate(self.layers):
      #if len(inputs.shape) != 1 and all(['dense' == l.mode for l in self.layers[layer_idx:]]):
      #  inputs = np.expand_dims(inputs[:,-1],axis=1)
      inputs = layer(inputs)
      
    return inputs

