import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.jit as jit
from .q_utils import *


class Dense(nn.Module):
    """
     dense layer
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, 
                     activation=None,frac_bits=32,int_bits=32,name=''):
        super(Dense, self).__init__()
        self.name = name
        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.bias          = bias 
        self.dropout       = dropout
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear.weight = self.init_weight(self.linear.weight)
        self.linear.bias = self.init_weight(self.linear.bias,initializer=nn.init.zeros_)
        self.act = activation
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        print('Dense layer',self.hidden_size,'neurons with ',int_bits,' ,',frac_bits,' bit precision')


    def init_weight(self,w,initializer= nn.init.xavier_uniform_):
        initializer(w)
        return w
        
    def forward(self, input,time_dim=1):
        # # input_ is of dimensionalty (batch, time_step, input_size, ...)
        if self.frac_bits < 32:
            weight = quantize_weight(self.linear.weight,self.int_bits,self.frac_bits)
            bias = quantize_weight(self.linear.bias,self.int_bits, self.frac_bits)
        else:
            weight = self.linear.weight
            bias = self.linear.bias

        #if len(input.shape) > 2 :
        #    outputs = [F.linear(x,weight,bias) for x in th.unbind(input, dim=time_dim)]
        #    outputs = th.stack(outputs,dim=time_dim)
        #else :
        outputs = F.linear(input,weight,bias if self.bias else None)

        outputs = self.act(outputs)
        self.states = outputs
        return outputs


