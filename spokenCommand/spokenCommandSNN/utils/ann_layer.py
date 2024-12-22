import numpy as np

class AnnLayer:

    def __init__(self, mode, size, ts_ann, Iin,
                 weights, bias, tau=None, ret_ratio=None,clip_relu=True):

        if mode == 'dense' and len(weights) == 2:
            raise ValueError(' Gave recurrent weights for a dense layer')
        if mode == 'lpRNN' and len(weights) == 1:
            raise ValueError(' Only input weights for a lpRNN layer')

        self.mode = mode
        self.size = size
        self.bias = bias
        self.weights = weights
        self.Iin = Iin
        self.ts_ann = ts_ann
        if tau != None:
            self.tau = tau
            self.ret_ratio = np.exp(-ts_ann/self.tau)

        elif ret_ratio != None:
            self.tau = -ts_ann / np.log(ret_ratio)
            self.ret_ratio = ret_ratio
            
        self.bias = self.bias
        self.clip_relu = clip_relu
        self.state = None
        self.probing = False

    def __call__(self,input):
        #input = (features,timestep)
        self.state = np.zeros((input.shape[1],self.size))
        x = np.zeros((self.size,)).astype(np.float32)
        for i in range(input.shape[1]):
            ip = input[:,i]
            if self.mode =='lpRNN':
                w_input = np.clip(np.dot(self.weights[0],ip) + np.dot(self.weights[1],x) + self.bias,0,self.Iin).astype(np.float32)
                x = (self.ret_ratio*x + (1-self.ret_ratio)*w_input).astype(np.float32) # filtering inside the neuron
            else:
                x = np.dot(self.weights[0],ip) + self.bias
                if self.clip_relu:
                    x = np.clip(x,0,self.Iin)

            self.state[i] = x

        return self.state.T.astype(np.float32)
              

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        r_string = 'Timestep ann:' +str(self.ts_ann)+ '\n'
        r_string += 'tau=' + str(self.tau) + '\n'
        r_string += 'feedforward weights' + str(self.weights[0]) + '\n'
        if self.mode == 'lpRNN':
            r_string += 'recurrent weights' + str(self.weights[1]) + '\n'
        r_string += '  '
        return r_string
