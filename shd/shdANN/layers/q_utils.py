import functools

def quantize_weight(weight,int_bits,frac_bits):
    upper = 2**(int_bits-1) - 1
    lower = -2**(int_bits-1) + 1
    scale = 2** frac_bits
    weight_q = weight.clamp(lower,upper).mul(scale).round().div(scale)
    weight_q = (weight_q - weight).detach() + weight
    return weight_q

'''
Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
'''
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

