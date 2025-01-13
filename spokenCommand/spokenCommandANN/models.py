from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.nn import LSTM
from layers import jit_lpRNN,jit_lpLinear,Dense
import torch

class ClampedRelu(nn.Module):
	def __init__(self,activation_max):
		super(ClampedRelu, self).__init__()
		self.activation_max = activation_max

	def forward(self,x):
		return torch.clamp(x,0,self.activation_max)

class small_lprnn_net(nn.Module):
	def __init__(self,nCategories,	
                    int_bits=32,
                    frac_bits=32,
                    input_size=80,
                    lpff_size=128,
                    nunits=64,
                    nlayers=2,
                    ret_ratio=0.8,
                    activation_max=6,
                    dev='cuda',
                    use_lstm=False,
                    use_bias=True):
	
		super().__init__()
		self.dev = dev
		self.num_layers = nlayers
		self.hidden_size = nunits
		self.retention_ratio = ret_ratio
		self.layers = []
		activation = ClampedRelu(activation_max)
		self.use_lstm = use_lstm
		if not self.use_lstm:
			self.lpff = jit_lpLinear(input_size=input_size,
									hidden_size=lpff_size,
									activation=activation,
									set_retention_ratio=self.retention_ratio,train_ret_ratio=False,bias=use_bias,int_bits=torch.tensor([int_bits]).to(dev),
									frac_bits=torch.tensor([frac_bits]).to(dev),name='lpff_1',device=dev)
		
		if use_lstm:
			rnn_layers = [LSTM(80,self.hidden_size,batch_first=True)]
			rnn_layers.extend([LSTM(self.hidden_size,self.hidden_size,batch_first=True) for i in range(self.num_layers-1)])

		else:
			rnn_layers = [jit_lpRNN(lpff_size,self.hidden_size,bias=use_bias,
					set_retention_ratio=self.retention_ratio,activation=activation,int_bits=torch.tensor([int_bits]).to(dev),frac_bits=torch.tensor([frac_bits]).to(dev),name='lprnn_1',device=dev)]
			rnn_layers.extend([jit_lpRNN(self.hidden_size,self.hidden_size,bias=use_bias,
					set_retention_ratio=self.retention_ratio,activation=activation,int_bits=torch.tensor([int_bits]).to(dev),frac_bits=torch.tensor([frac_bits]).to(dev),name='lprnn_'+str(i+2),device=dev) for i in range(self.num_layers-1)])
						
		self.rnn_layers = nn.ModuleList(rnn_layers)

		self.out = Dense(self.hidden_size,nCategories,activation=nn.Identity(),int_bits=int_bits,frac_bits=frac_bits,name='output',bias=use_bias)
	
	def output(self,x):
		x = self.out(x)
		x = torch.mean(x,dim=1)
		return x
	
	def quantize(self):
		if not self.use_lstm:
			self.lpff.quantize()
			for i in range(self.num_layers):
				self.rnn_layers[i].quantize()

	def forward(self, x):
		x = x.float()
		self.quantize()
		# batch timestep features 
		self.states = []
		if not self.use_lstm:
			x = self.lpff(x)
			self.states.append(x)
		for i in range(len(self.rnn_layers)):
			x = self.rnn_layers[i](x)
			if self.use_lstm:
				x, _ = x
			self.states.append(x)
		#x = x[:,-1]
		x = self.output(x)
		return x
