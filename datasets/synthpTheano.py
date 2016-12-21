'''
Parameters for synthetic examples using pykalman

(For linear models specify all, non-linear, just the ones that are relevant)
'''
import theano
import theano.tensor as T
import numpy as np
import os
from utils.misc import loadHDF5
def linear_trans(z,fxn_params = {}, ns=None): 
    return z+0.05
def linear_obs(z,fxn_params = {}, ns=None): 
    return 0.5*z
def nlinear_trans(z, fxn_params = {}, ns=None): 
    return 2*T.sin(z)+z

def nlinear_trans_learn(z, fxn_params = {}, ns = None):
    assert z.ndim == 3,'expecting 3d'
    z_1 = z[:,:,[0]]
    z_2 = z[:,:,[1]]
    f_1 = 0.2*z_1+T.tanh(fxn_params['alpha']*z_2)
    f_2 = 0.2*z_2+T.sin(fxn_params['beta']*z_1)
    return T.concatenate([f_1,f_2],axis=2)

def obs_learn(z,fxn_params = {}, ns = None):
    assert z.ndim == 3,'expecting 3d'
    return 0.5*z 

#Load saved matrices and use them to form theano transition and emission functions
SAVEDIR        = os.path.dirname(os.path.realpath(__file__))+'/synthetic'
saved_matrices   = loadHDF5(SAVEDIR+'/linear-matrices.h5')
saved_matrices_2 = loadHDF5(SAVEDIR+'/linear-matrices-2.h5')


def linear_trans_s12(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices['Wtrans_10'].astype('float32'),name = 'Wtrans_10')
    b   = theano.shared(saved_matrices['btrans_10'].astype('float32'),name = 'btrans_10')
    return T.dot(z,W)+b
def linear_obs_s12(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices['Wobs_10'].astype('float32'),name = 'Wobs_10')
    return T.dot(z,W)

def linear_trans_s13(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices['Wtrans_100'].astype('float32'),name = 'Wtrans_100')
    b   = theano.shared(saved_matrices['btrans_100'].astype('float32'),name = 'btrans_100')
    return T.dot(z,W)+b
def linear_obs_s13(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices['Wobs_100'].astype('float32'),name = 'Wobs_100')
    return T.dot(z,W)

def linear_trans_s14(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices['Wtrans_250'].astype('float32'),name = 'Wtrans_250')
    b   = theano.shared(saved_matrices['btrans_250'].astype('float32'),name = 'btrans_250')
    return T.dot(z,W)+b
def linear_obs_s14(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices['Wobs_250'].astype('float32'),name = 'Wobs_250')
    return T.dot(z,W)

def linear_trans_s15(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wtrans_10'].astype('float32'),name = 'Wtrans_10')
    b   = theano.shared(saved_matrices_2['btrans_10'].astype('float32'),name = 'btrans_10')
    return T.dot(z,W)+b
def linear_obs_s15(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wobs_10'].astype('float32'),name = 'Wobs_10')
    return T.dot(z,W)
def linear_trans_s16(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wtrans_100'].astype('float32'),name = 'Wtrans_100')
    b   = theano.shared(saved_matrices_2['btrans_100'].astype('float32'),name = 'btrans_100')
    return T.dot(z,W)+b
def linear_obs_s16(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wobs_100'].astype('float32'),name = 'Wobs_100')
    return T.dot(z,W)
def linear_trans_s17(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wtrans_250'].astype('float32'),name = 'Wtrans_250')
    b   = theano.shared(saved_matrices_2['btrans_250'].astype('float32'),name = 'btrans_250')
    return T.dot(z,W)+b
def linear_obs_s17(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wobs_250'].astype('float32'),name = 'Wobs_250')
    return T.dot(z,W)

def linear_trans_s18(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wtrans_10_diag'].astype('float32'),name = 'Wtrans_10_diag')
    b   = theano.shared(saved_matrices_2['btrans_10_diag'].astype('float32'),name = 'btrans_10_diag')
    return T.dot(z,W)+b
def linear_obs_s18(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wobs_10_diag'].astype('float32'),name = 'Wobs_10_diag')
    return T.dot(z,W)
def linear_trans_s19(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wtrans_100_diag'].astype('float32'),name = 'Wtrans_100_diag')
    b   = theano.shared(saved_matrices_2['btrans_100_diag'].astype('float32'),name = 'btrans_100_diag')
    return T.dot(z,W)+b
def linear_obs_s19(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wobs_100_diag'].astype('float32'),name = 'Wobs_100_diag')
    return T.dot(z,W)
def linear_trans_s20(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wtrans_250_diag'].astype('float32'),name = 'Wtrans_250_diag')
    b   = theano.shared(saved_matrices_2['btrans_250_diag'].astype('float32'),name = 'btrans_250_diag')
    return T.dot(z,W)+b
def linear_obs_s20(z,fxn_params = {},ns=None): 
    assert z.ndim==3,'Expecting 3d'
    W   = theano.shared(saved_matrices_2['Wobs_250_diag'].astype('float32'),name = 'Wobs_250_diag')
    return T.dot(z,W)

def updateParamsSynthetic(params_synthetic):
    params_synthetic['synthetic9']['trans_fxn']   = linear_trans
    params_synthetic['synthetic9']['obs_fxn']     = linear_obs

    params_synthetic['synthetic10']['trans_fxn']  = nlinear_trans
    params_synthetic['synthetic10']['obs_fxn']    = linear_obs

    params_synthetic['synthetic11']['trans_fxn']  = nlinear_trans_learn
    params_synthetic['synthetic11']['obs_fxn']    = obs_learn

    params_synthetic['synthetic12']['trans_fxn']  = linear_trans_s12
    params_synthetic['synthetic12']['obs_fxn']    = linear_obs_s12

    params_synthetic['synthetic13']['trans_fxn']  = linear_trans_s13
    params_synthetic['synthetic13']['obs_fxn']    = linear_obs_s13

    params_synthetic['synthetic14']['trans_fxn']  = linear_trans_s14
    params_synthetic['synthetic14']['obs_fxn']    = linear_obs_s14

    params_synthetic['synthetic15']['trans_fxn']  = linear_trans_s15
    params_synthetic['synthetic15']['obs_fxn']    = linear_obs_s15

    params_synthetic['synthetic16']['trans_fxn']  = linear_trans_s16
    params_synthetic['synthetic16']['obs_fxn']    = linear_obs_s16

    params_synthetic['synthetic17']['trans_fxn']  = linear_trans_s17
    params_synthetic['synthetic17']['obs_fxn']    = linear_obs_s17

    params_synthetic['synthetic18']['trans_fxn']  = linear_trans_s18
    params_synthetic['synthetic18']['obs_fxn']    = linear_obs_s18

    params_synthetic['synthetic19']['trans_fxn']  = linear_trans_s19
    params_synthetic['synthetic19']['obs_fxn']    = linear_obs_s19

    params_synthetic['synthetic20']['trans_fxn']  = linear_trans_s20
    params_synthetic['synthetic20']['obs_fxn']    = linear_obs_s20
