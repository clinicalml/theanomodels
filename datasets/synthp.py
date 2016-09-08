'''
Parameters for synthetic examples using pykalman
(For linear models specify all, non-linear, just the ones that are relevant)
'''
import numpy as np
def nlinear_trans(z,fxn_params = {}, ns=None): 
    return 2*np.sin(z)+z
def linear_trans(z,fxn_params = {},ns=None): 
    return z+0.05
def linear_obs(z,fxn_params = {},ns=None): 
    return 0.5*z
def nlinear_trans_learn(z, fxn_params = {}, ns=None): 
    return 2*np.sin(fxn_params['alpha']*z)+z

params_synthetic = {}
params_synthetic['synthetic9'] = {}
params_synthetic['synthetic9']['trans_fxn']   = linear_trans
params_synthetic['synthetic9']['obs_fxn']     = linear_obs
params_synthetic['synthetic9']['params']      = {}
params_synthetic['synthetic9']['trans_cov']   = 10.
params_synthetic['synthetic9']['trans_drift'] = 0.05
params_synthetic['synthetic9']['trans_mult']  = 1.
params_synthetic['synthetic9']['obs_cov']     = 20. 
params_synthetic['synthetic9']['obs_drift']   = 0. 
params_synthetic['synthetic9']['obs_mult']    = 0.5 
params_synthetic['synthetic9']['init_mu']     = 0.
params_synthetic['synthetic9']['init_cov']    = 1.
params_synthetic['synthetic9']['baseline']    = 'KF' 
params_synthetic['synthetic9']['docstr']      = '$z_t\sim\mathcal{N}(z_{t-1}+0.05,10)$\n$x_t\sim\mathcal{N}(0.5z_t,20)$'


params_synthetic['synthetic10'] = {}
params_synthetic['synthetic10']['trans_fxn']   = nlinear_trans
params_synthetic['synthetic10']['obs_fxn']     = linear_obs
params_synthetic['synthetic10']['params']      = {}
params_synthetic['synthetic10']['trans_cov']   = 5.
params_synthetic['synthetic10']['trans_drift'] = 0.
params_synthetic['synthetic10']['trans_mult']  = 1.
params_synthetic['synthetic10']['obs_cov']     = 5. 
params_synthetic['synthetic10']['obs_drift']   = 0. 
params_synthetic['synthetic10']['obs_mult']    = 0.5 
params_synthetic['synthetic10']['init_mu']     = 0.
params_synthetic['synthetic10']['init_cov']    = 0.01 
params_synthetic['synthetic10']['baseline']    = 'UKF' 
params_synthetic['synthetic10']['docstr']       = '$z_t\sim\mathcal{N}(2\sin(z_{t-1})+z_{t-1},5)$\n$x_t\sim\mathcal{N}(0.5z_t,5)$'

params_synthetic['synthetic11'] = {}
params_synthetic['synthetic11']['params']      = {}
params_synthetic['synthetic11']['params']['alpha'] = 2.5
params_synthetic['synthetic11']['trans_fxn']   = nlinear_trans_learn
params_synthetic['synthetic11']['obs_fxn']     = linear_obs
params_synthetic['synthetic11']['trans_cov']   = 5.
params_synthetic['synthetic11']['trans_drift'] = 0.
params_synthetic['synthetic11']['trans_mult']  = 1.
params_synthetic['synthetic11']['obs_cov']     = 5. 
params_synthetic['synthetic11']['obs_drift']   = 0. 
params_synthetic['synthetic11']['obs_mult']    = 0.5 
params_synthetic['synthetic11']['init_mu']     = 0.
params_synthetic['synthetic11']['init_cov']    = 0.01 
params_synthetic['synthetic11']['baseline']    = 'UKF' 
params_synthetic['synthetic11']['docstr']       = '$z_t\sim\mathcal{N}(2\sin(\alpha z_{t-1})+z_{t-1},5)$\n$x_t\sim\mathcal{N}(0.5z_t,5)$'
