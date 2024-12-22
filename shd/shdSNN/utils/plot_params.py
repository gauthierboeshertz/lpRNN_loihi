import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap



def find_params(tau_prev,lowest_wann,Iin,mode='lpRNN',decay_bounds =(40,100),wfb_bounds=(1,2048)):
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

    mapping = dict()
    if mode == 'lpRNN' :
        for decay in range(decay_bounds[0], decay_bounds[1]):
            ts = decay*tau_prev/4096
            taus = tau_prev/ts
            for wfb in range(wfb_bounds[0], wfb_bounds[1]):
                f =  wfb*taus*64/(Iin)
                if f > 2**23/(2*Iin):
                    continue
                    
                wsn = wfb*lowest_wann/taus 
                mapping[(decay,wfb)] = (float(wsn),ts,f)
                """
                if not check_mantexp(wfb):
                    mant_,exp_ = get_mantexp(wfb)
                    Wfb_r = int(mant_)*2**exp_
                    f_r = Wfb_r *taus*64/Iin
                    if f_r > 2**23/(2*Iin):
                        continue
                    wsn_r = Wfb_r*lowest_wann/taus 
                    mapping[(decay,wfb)] = (float(wsn_r),ts)
                """
    else :
        for decay in range(decay_bounds[0], decay_bounds[1]):
            ts = decay*tau_prev/4096
            taus = tau_prev/ts

            for wfb in range(wfb_bounds[0], wfb_bounds[1]):
                f =  wfb*taus*64/(Iin)
                if f >= 2**23/Iin:
                    continue
                wsn = wfb*lowest_wann
                mapping[(decay,wfb)] = (float(wsn),ts,f)
                """
                if not check_mantexp(wfb):
                    mant_,exp_ = get_mantexp(wfb)
                    Wfb_r = int(mant_)*2**exp_
                    f_r = Wfb_r *taus*64/Iin
                    if f_r >=2**23/(2*Iin):
                        continue
                    wsn_r = Wfb_r*lowest_wann
                    mapping[(decay,wfb)] = (float(wsn_r),ts)
                """
    return mapping

def plot_params(tau,Iin,lowest_wann=None,frac_bits = None,mode='lpRNN',decay_bounds =(40,100),wfb_bounds=(1,2048)):

    if lowest_wann is None and frac_bits is None:
        raise Exception('Both lowest ann and frac_bits are none')

    if (not (lowest_wann is None)) and (not (frac_bits is  None)):
        raise Exception('Both lowest ann and frac_bits have a value')

    if lowest_wann is None:
        lowest_wann = 2**(-frac_bits)
    mapping = find_params(tau,lowest_wann,Iin,mode=mode,decay_bounds=decay_bounds,wfb_bounds=wfb_bounds)
    decay = np.arange(1,decay_bounds[1],1)
    wfb = np.arange(1, wfb_bounds[1], 1)

    maps = []
    zero_points = []
    for dec in decay:
        m = []
        for wf in wfb:
            if (dec,wf)  in mapping.keys():
                m.append(mapping[(dec,wf)][0])
                if mapping[(dec,wf)][0]%1 <= 1e-5 and mapping[(dec,wf)][0] >=0.5 :
                    zero_points.append(((dec,wf),mapping[(dec,wf)][1],mapping[(dec,wf)][0],mapping[(dec,wf)][2]))
            else:
                m.append(-1)
        m = np.array(m)
        maps.append(m)
    
    maps = np.stack(maps)
    maps = maps.T
    maps_difference = np.abs(maps - np.round(maps))
    #maps_difference[maps == -1 ] = -1
    fig, ax = plt.subplots() 
    
    # Plot the surface.
    ax.set_xlabel('decay') 
    ax.set_ylabel('feedback weight') 
    ax.set_title("Error when rounding the SNN weights") 
    img = ax.imshow(maps_difference,aspect='auto')
    for p in zero_points:
        print('possible timesteps :',p[1],'value of weight:',p[2],'decay:',p[0][0],'factor:',p[3],'wfb:',p[0][1])
        ax.scatter([p[0][0]],[p[0][1]],c='r',label = 'possible points')
    # Customize the z axis.
    
    fig.colorbar(img,ax=ax)
    plt.show()
    fig.savefig("weights_error")

                    


    
    
    
