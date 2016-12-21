import numpy as np
from synthp import params_synthetic
from synthpTheano import updateParamsSynthetic
import theano
import theano.tensor as T
for ds, sdata in zip([10,100,250],['12','13','14']):
    dset = 'synthetic'+sdata
    zinp = np.random.randn(3,10,ds).astype('float32') 
    print 'Transition: ',params_synthetic[dset]['trans_fxn'](zinp).shape
    print 'Obs: ',params_synthetic[dset]['obs_fxn'](zinp).shape
updateParamsSynthetic(params_synthetic)
print 'Theano...'
for ds, sdata in zip([10,100,250],['12','13','14']):
    dset = 'synthetic'+sdata
    zinp = theano.shared(np.random.randn(3,10,ds).astype('float32')) 
    print 'Transition: ',params_synthetic[dset]['trans_fxn'](zinp).eval().shape
    print 'Obs: ',params_synthetic[dset]['obs_fxn'](zinp).eval().shape
