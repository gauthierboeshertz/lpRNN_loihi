from brian2 import Equations
import warnings

# neuron eq with used for spiking input

loihi_eq = Equations('''

di_ip/dt = -i_ip/(tau_ip*dt) + i_in /(dt)   : 1

di_in/dt = -i_in/(tau_in*dt)   + bias/dt : 1

dv/dt = -v/(tau_v*dt)+  (1/dt)*(i_ip+ i_fb) : 1 (unless refractory)

di_fb/dt = -i_fb/(tau_fb*dt) : 1

vThMant : 1 (shared)

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


def add_rounding_to_NeuronGroup(neurongroup):

    round_v = 'v = int(v)'
    round_i_ip = 'i_ip = int(i_ip)'
    round_i_in = 'i_in = int(i_in)'
    round_i_fb = 'i_fb = int(i_fb)'

    reg_round_v = neurongroup.run_regularly(round_v)
    reg_round_fb = neurongroup.run_regularly(round_i_fb)
    reg_round_ip = neurongroup.run_regularly(round_i_ip)
    reg_round_in = neurongroup.run_regularly(round_i_in)

    return [reg_round_v,reg_round_fb,reg_round_ip,reg_round_in]

def add_clipping_to_NeuronGroup(neurongroup):
    clip_v = 'v = clip(v,0,2**23)'
    clip_i_fb = 'i_fb = clip(i_fb,-2**23,2**23)'
    clip_i_ip = 'i_ip = clip(i_ip,0,2**23)'
    clip_i_in = 'i_in = clip(i_in,-2**23,2**23)'

    reg_clip_v = neurongroup.run_regularly(clip_v)
    reg_clip_fb = neurongroup.run_regularly(clip_i_fb)
    reg_clip_ip = neurongroup.run_regularly(clip_i_ip)
    reg_clip_in = neurongroup.run_regularly(clip_i_in)

    return [reg_clip_v,reg_clip_fb,reg_clip_ip,reg_clip_in]


def add_relu_to_NeuronGroup(neurongroup):
    clip_v = 'v = (v+abs(v))/2'
    clip_i_ip = 'i_ip = (i_ip + abs(i_ip))/2'

    reg_clip_v = neurongroup.run_regularly(clip_v)
    reg_clip_ip = neurongroup.run_regularly(clip_i_ip)

    return [reg_clip_v,reg_clip_ip]


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
