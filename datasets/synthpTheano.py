'''
Parameters for synthetic examples using pykalman

(For linear models specify all, non-linear, just the ones that are relevant)
'''
import theano
import theano.tensor as T
import numpy as np
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

def updateParamsSynthetic(params_synthetic):
    params_synthetic['synthetic9']['trans_fxn']   = linear_trans
    params_synthetic['synthetic9']['obs_fxn']     = linear_obs

    params_synthetic['synthetic10']['trans_fxn']  = nlinear_trans
    params_synthetic['synthetic10']['obs_fxn']    = linear_obs

    params_synthetic['synthetic11']['trans_fxn']  = nlinear_trans_learn
    params_synthetic['synthetic11']['obs_fxn']    = obs_learn
