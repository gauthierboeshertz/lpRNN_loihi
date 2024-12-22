from brian2 import Equations
import warnings

# neuron eq with used for spiking input

loihi_eq = Equations('''

dv/dt = -v/(tau_v*dt)+  (1/dt)*(i_ip+ i_fb) : 1 (unless refractory)
di_fb/dt = -i_fb/(tau_fb*dt) : 1

di_ip/dt = -i_ip/(tau_ip*dt) + i_in /(dt)+ i_rec/(dt)  + bias/(dt): 1

di_in/dt = -i_in/(tau_in*dt)   : 1
di_rec/dt = -i_rec/(tau_rec*dt)   : 1


vThMant : 1 (shared)
tau_rec  : 1 (shared)
tau_v  : 1 (shared)
tau_in :1 (shared)
tau_ip  : 1 (shared)
tau_fb  : 1 (shared)
bias :1 
gain :1 (shared)

threshold = vThMant * 64 : 1
ref_p : 1 (shared)
''')


loihi_eq_dict = {
    'model': loihi_eq,
    'threshold': 'v>threshold',
    'reset': 'v = 0',
    'refractory': 'ref_p*dt',
    'method': 'euler',
}


loihi_eq_dense = Equations('''
dv/dt = -v/(tau_v*dt)+  (1/dt)*(i_ip+ i_fb) + bias/(dt): 1 (unless refractory)
di_fb/dt = -i_fb/(tau_fb*dt) : 1

di_ip/dt = -i_ip/(tau_ip*dt) + i_in /(dt)+ i_rec/(dt)  : 1

di_in/dt = -i_in/(tau_in*dt)   : 1
di_rec/dt = -i_rec/(tau_rec*dt)   : 1


vThMant : 1 (shared)
tau_rec  : 1 (shared)
tau_v  : 1 (shared)
tau_in :1 (shared)
tau_ip  : 1 (shared)
tau_fb  : 1 (shared)
bias :1 
gain :1 (shared)

threshold = vThMant * 64 : 1
ref_p : 1 (shared)
''')


loihi_eq_dict_dense = {
    'model': loihi_eq,
    'threshold': 'v>threshold',
    'reset': 'v = 0',
    'refractory': 'ref_p*dt',
    'method': 'euler',
}


# synapse eq
syn_eq = '''weight :1
w_factor : 1 (shared)'''
on_pre = '''
i_fb_post += 64 * weight * w_factor
'''


loihi_syn_dict = {'model': syn_eq,
                  'on_pre': on_pre,
                  'method': 'euler'
                  }



def add_clipping_to_NeuronGroup(neurongroup):
    clip_v = 'v = clip(v,0,2**23)'
    clip_ip = 'i_ip = clip(i_ip,0,2**23)'

    neurongroup.run_regularly(clip_v)
    neurongroup.run_regularly(clip_ip)



def set_params(briangroup, params, ndargs=None):
    for par in params:
        if hasattr(briangroup, par):
            if ndargs is not None and par in ndargs:
                if ndargs[par] is None:
                    setattr(briangroup, par, params[par])
                else:
                    print(par, ndargs, ndargs[par])
                    setattr(briangroup, par, ndargs[par])
            else:
                setattr(briangroup, par, params[par])
        else:
            warnings.warn("Group " + str(briangroup.name) +
                          " has no state variable " + str(par) +
                          ", but you tried to set it with set_params")