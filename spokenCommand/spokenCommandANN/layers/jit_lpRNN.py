from layers.q_utils import quantize_weight
import torch
import torch.nn as nn
import torch.jit as jit
from torch import Tensor
import torch.nn.functional as F


"""
jit version created a 20% faster implementation
"""
from layers.q_utils import *
"""
jit version created a 20% faster implementation
"""
class jit_lpRNN(jit.ScriptModule):
    """
    An implementation of Elman RNN with dropout, weight dropout and low pass filtering added:
    retention_ratio: for low pass filtering the RNN
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 bias=True, 
                 dropout=0.0,
                 train_ret_ratio=False, 
                 set_retention_ratio=None,
                 activation=nn.ReLU(),
                 int_bits=32,
                 frac_bits=8,
                 device='cuda',
                 name='',
                 time_dim=1):
        super(jit_lpRNN, self).__init__()
        self.time_dim = time_dim
        self.name = name
        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.bias          = bias
        self.dropout       = dropout
        self.train_ret_ratio = train_ret_ratio > 0
        self.act = activation
        self.device = device
        self.linear = nn.Linear(input_size, hidden_size, bias=True).to(self.device)
        self.linear_hh = nn.Linear(hidden_size, hidden_size, bias=False).to(self.device) 
        self.linear.weight = self.init_weight(self.linear.weight).to(self.device)
        self.linear.bias = self.init_weight(self.linear.bias,initializer=nn.init.zeros_).to(self.device)
        self.linear_hh.weight = self.init_weight(self.linear_hh.weight,initializer=nn.init.orthogonal_).to(self.device)
        
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        print('lpRNN layer',self.hidden_size,'neurons with ',int_bits,' ,',frac_bits,' bit precision')
        # Recurrent activation
        # Train low pass filtering factor
        if set_retention_ratio is not None:
            self.retention_ratio = nn.Parameter(set_retention_ratio * torch.ones(self.hidden_size)
                                                ,requires_grad=self.train_ret_ratio).to(self.device)
        else:
            self.retention_ratio = nn.Parameter(torch.FloatTensor(self.hidden_size).uniform_(0.01, 1)
                                                ,requires_grad=self.train_ret_ratio).to(self.device)

        self.quantize()
        
    def init_weight(self,w,initializer= nn.init.xavier_uniform_):
        initializer(w)
        return w

    def quantize(self):
        if self.frac_bits < 32:
            self.q_weight = quantize_weight(self.linear.weight,self.int_bits,self.frac_bits)
            if self.bias:
                self.q_bias = quantize_weight(self.linear.bias,self.int_bits, self.frac_bits)
            else:
                self.q_bias = None

            self.q_weight_hh = quantize_weight(self.linear_hh.weight,self.int_bits, self.frac_bits)

        else:
            self.q_weight = self.linear.weight
            self.q_bias = self.linear.bias
            self.q_weight_hh = self.linear_hh.weight


    @torch.jit.script_method
    def forward(self, 
                ip):
        
        # type: (Tensor) -> Tensor
         
        hx = torch.zeros(1,self.hidden_size)
        hx = hx.to(ip.device)
        outputs = torch.zeros((ip.shape[0],ip.shape[1],self.hidden_size)).to(self.device)
        for time_idx,x in enumerate(torch.unbind(ip, dim=self.time_dim)):
            if self.bias:
                hy = F.linear(x,self.q_weight, self.q_bias) + torch.mm(hx, self.q_weight_hh.t())
            else:
                hy = F.linear(x,self.q_weight, None) + torch.mm(hx, self.q_weight_hh.t())
            hy = self.act(hy)
            # Filtering 
            hx = self.retention_ratio * hx + (1-self.retention_ratio) * hy
            outputs[:,time_idx,:] += hx

        return outputs



