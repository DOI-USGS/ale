import numpy as np

def check_key(f):
    def wrapper(*args):
        obj = args[0]  # This is the class instance; self
        key = args[1]  # We assume the first arg is the key
        assert key in obj.pool_keys
        return f(*args)
    return wrapper

class SimpleSpice():
    def __init__(self, pool_keys=[]):
        base_keys = ['INS-12345_CCD_CENTER',
                     'INS-12345_FOCAL_LENGTH',
                     'INS-12345_ITRANSL',
                     'INS-12345_ITRANSS',
                     'INSTRUMENT_NAME',
                     'SPACECRAFT_NAME']
        self.pool_keys = pool_keys + base_keys

    def scs2e(self, *args):
        return 0.1
    
    def bods2c(self, x):
        return -12345
    
    @check_key
    def gdpool(self, key, x, length):
        return np.ones(length)
    
    @check_key
    def bodvrd(self, key, x, length):
        return (3, np.ones(length,))
    
    def spkpos(self, *args):
        return (np.ones(3), None)
    
    def spkezr(self, *args):
        return (np.ones(6), None)
    
    def furnsh(self, *args):
        return
    
    def unload(self, *args):
        return
    
    def pxform(self, *args):
        return
    
    def m2q(self, *args):
        return np.asarray([1,2,3,4])
    
    def bodn2c(self, *args):
        return "SPACE"

def get_mockkernels(self, *args):
    return "some_metakernel"
