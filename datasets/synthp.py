'''
Parameters for synthetic examples using pykalman
(For linear models specify all, non-linear, just the ones that are relevant)
'''
import numpy as np
import os
from utils.misc import loadHDF5,saveHDF5
def nlinear_trans(z,fxn_params = {}, ns=None): 
    return 2*np.sin(z)+z
def linear_trans(z,fxn_params = {},ns=None): 
    return z+0.05
def linear_obs(z,fxn_params = {},ns=None): 
    return 0.5*z

params_synthetic = {}
params_synthetic['synthetic9'] = {}
params_synthetic['synthetic9']['trans_fxn']   = linear_trans
params_synthetic['synthetic9']['obs_fxn']     = linear_obs
params_synthetic['synthetic9']['dim_obs']     = 1
params_synthetic['synthetic9']['dim_stoc']    = 1
params_synthetic['synthetic9']['params']      = {}
params_synthetic['synthetic9']['trans_cov']   = 10.
params_synthetic['synthetic9']['trans_cov_full']   = 10.
params_synthetic['synthetic9']['trans_drift'] = 0.05
params_synthetic['synthetic9']['trans_mult']  = 1.
params_synthetic['synthetic9']['obs_cov']     = 20. 
params_synthetic['synthetic9']['obs_cov_full']     = 20. 
params_synthetic['synthetic9']['obs_drift']   = 0. 
params_synthetic['synthetic9']['obs_mult']    = 0.5 
params_synthetic['synthetic9']['init_mu']     = 0.
params_synthetic['synthetic9']['init_cov']    = 1.
params_synthetic['synthetic9']['init_cov_full']    = 1.
params_synthetic['synthetic9']['baseline']    = 'KF' 
params_synthetic['synthetic9']['docstr']      = '$z_t\sim\mathcal{N}(z_{t-1}+0.05,10)$\n$x_t\sim\mathcal{N}(0.5z_t,20)$'

params_synthetic['synthetic10'] = {}
params_synthetic['synthetic10']['trans_fxn']   = nlinear_trans
params_synthetic['synthetic10']['obs_fxn']     = linear_obs
params_synthetic['synthetic10']['dim_obs']     = 1
params_synthetic['synthetic10']['dim_stoc']    = 1
params_synthetic['synthetic10']['params']      = {}
params_synthetic['synthetic10']['trans_cov']   = 5.
params_synthetic['synthetic10']['trans_cov_full']   = 5.
params_synthetic['synthetic10']['trans_drift'] = 0.
params_synthetic['synthetic10']['trans_mult']  = 1.
params_synthetic['synthetic10']['obs_cov']     = 5. 
params_synthetic['synthetic10']['obs_cov_full']= 5. 
params_synthetic['synthetic10']['obs_drift']   = 0. 
params_synthetic['synthetic10']['obs_mult']    = 0.5 
params_synthetic['synthetic10']['init_mu']     = 0.
params_synthetic['synthetic10']['init_cov']    = 0.01 
params_synthetic['synthetic10']['init_cov_full']= 0.01 
params_synthetic['synthetic10']['baseline']    = 'UKF' 
params_synthetic['synthetic10']['docstr']      = '$z_t\sim\mathcal{N}(2\sin(z_{t-1})+z_{t-1},5)$\n$x_t\sim\mathcal{N}(0.5z_t,5)$'

def nlinear_trans_learn(z, fxn_params = {}, ns=None): 
    assert z.ndim == 3,'expecting 3d'
    assert z.shape[2]== 2,'expecting 2 dim'
    z_1 = z[:,:,[0]]
    z_2 = z[:,:,[1]]
    f_1 = 0.2*z_1+np.tanh(fxn_params['alpha']*z_2)
    f_2 = 0.2*z_2+np.sin(fxn_params['beta']*z_1)
    return np.concatenate([f_1,f_2],axis=2)
def obs_learn(z,fxn_params = {}, ns = None):
    assert z.ndim == 3,'expecting 3d'
    return 0.5*z
params_synthetic['synthetic11']                = {}
params_synthetic['synthetic11']['params']      = {}
params_synthetic['synthetic11']['params']['alpha']  = 0.5
params_synthetic['synthetic11']['params']['beta']  = -0.1
params_synthetic['synthetic11']['trans_fxn']   = nlinear_trans_learn
params_synthetic['synthetic11']['obs_fxn']     = obs_learn 
params_synthetic['synthetic11']['dim_obs']     = 2
params_synthetic['synthetic11']['dim_stoc']    = 2
params_synthetic['synthetic11']['trans_cov']   = 1. 
params_synthetic['synthetic11']['trans_cov_full']   = None
params_synthetic['synthetic11']['trans_drift'] = 0.
params_synthetic['synthetic11']['trans_mult']  = 1.
params_synthetic['synthetic11']['obs_cov']     = 0.1 
params_synthetic['synthetic11']['obs_cov_full']     = None
params_synthetic['synthetic11']['obs_drift']   = 0. 
params_synthetic['synthetic11']['obs_mult']    = 0.5 
params_synthetic['synthetic11']['init_mu']     = 0.
params_synthetic['synthetic11']['init_cov']    = 0.01 
params_synthetic['synthetic11']['init_cov_full']    = None
params_synthetic['synthetic11']['baseline']    = 'None' 
params_synthetic['synthetic11']['docstr']      = '$z_t\sim\mathcal{N}([0.2z_{t-1}^0+\\text{tanh}(\\alpha z_{t-1}^1); 0.2z_{t-1}^1+\\sin(\\beta z_{t-1}^0)] ,'+str(params_synthetic['synthetic11']['trans_cov'])+')$\n$x_t\sim\mathcal{N}(0.5z_t,'+str(params_synthetic['synthetic11']['obs_cov'])+')$'

"""
Synthetic [12,13,14] to check scalability of the inference algorithm
"""

SAVEDIR = os.path.dirname(os.path.realpath(__file__))+'/synthetic'
if not os.path.exists(SAVEDIR+'/linear-matrices.h5'):
    os.system('mkdir -p '+SAVEDIR)
    print 'Creating linear matrices'
    linmat = {}
    np.random.seed(0)
    linmat['Wtrans_10']  = np.random.randn(10,10)*0.05
    linmat['Wtrans_100'] = np.random.randn(100,100)*0.05
    linmat['Wtrans_250'] = np.random.randn(250,250)*0.05
    linmat['btrans_10']  = np.random.randn(10,)*0.05
    linmat['btrans_100'] = np.random.randn(100,)*0.05
    linmat['btrans_250'] = np.random.randn(250,)*0.05
    linmat['Wobs_10']  = np.random.randn(10,20)*0.05
    linmat['Wobs_100'] = np.random.randn(100,200)*0.05
    linmat['Wobs_250'] = np.random.randn(250,500)*0.05
    saveHDF5(SAVEDIR+'/linear-matrices.h5',linmat)
    saved_matrices = linmat
else:
    print 'Loading linear matrices'
    saved_matrices = loadHDF5(SAVEDIR+'/linear-matrices.h5')

if not os.path.exists(SAVEDIR+'/linear-matrices-2.h5'):
    os.system('mkdir -p '+SAVEDIR)
    print 'Creating linear matrices'
    linmat = {}
    np.random.seed(0)
    linmat['Wtrans_10']  = np.random.randn(10,10)*0.05
    linmat['Wtrans_100'] = np.random.randn(100,100)*0.05
    linmat['Wtrans_250'] = np.random.randn(250,250)*0.05
    linmat['btrans_10']  = np.random.randn(10,)*0.05
    linmat['btrans_100'] = np.random.randn(100,)*0.05
    linmat['btrans_250'] = np.random.randn(250,)*0.05
    linmat['Wobs_10']  = np.random.randn(10,10)*0.05
    linmat['Wobs_100'] = np.random.randn(100,100)*0.05
    linmat['Wobs_250'] = np.random.randn(250,250)*0.05
    #Diagonal
    def setupMatrices(dim, mult=0.05):
        di  = np.diag_indices(dim)
        bt  = np.random.randn(dim,)*mult
        Wt  = np.zeros((dim,dim))
        Wt[di] = np.random.randn(dim,)*mult
        We  = np.zeros((dim,dim))
        We[di] = np.random.randn(dim,)*mult
        return Wt,bt,We
    linmat['Wtrans_10_diag'], linmat['btrans_10_diag'], linmat['Wobs_10_diag']  = setupMatrices(10)
    linmat['Wtrans_100_diag'],linmat['btrans_100_diag'],linmat['Wobs_100_diag'] = setupMatrices(100)
    linmat['Wtrans_250_diag'],linmat['btrans_250_diag'],linmat['Wobs_250_diag'] = setupMatrices(250)
    #Check that this works
    import ipdb;ipdb.set_trace()
    saveHDF5(SAVEDIR+'/linear-matrices-2.h5',linmat)
    saved_matrices_2= linmat
else:
    print 'Loading linear matrices'
    saved_matrices_2= loadHDF5(SAVEDIR+'/linear-matrices-2.h5')

params_synthetic['synthetic12'] = {}
params_synthetic['synthetic12']['dim_obs']     = 20
params_synthetic['synthetic12']['dim_stoc']    = 10
params_synthetic['synthetic12']['params']      = {}

params_synthetic['synthetic12']['trans_cov']   = 10.
params_synthetic['synthetic12']['trans_cov_full']   = np.zeros((10,10))
np.fill_diagonal(params_synthetic['synthetic12']['trans_cov_full'], 10.)

params_synthetic['synthetic12']['trans_drift'] = saved_matrices['btrans_10'] 
params_synthetic['synthetic12']['trans_mult']  = saved_matrices['Wtrans_10']

params_synthetic['synthetic12']['obs_cov']     = 20.
params_synthetic['synthetic12']['obs_cov_full']= np.zeros((20,20))
np.fill_diagonal(params_synthetic['synthetic12']['obs_cov_full'], 20.)

params_synthetic['synthetic12']['obs_drift']   = np.ones((20,))*0. 
params_synthetic['synthetic12']['obs_mult']    = saved_matrices['Wobs_10'] 
def linear_trans_s12(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices['Wtrans_10'])+saved_matrices['btrans_10']
def linear_obs_s12(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices['Wobs_10'])
params_synthetic['synthetic12']['trans_fxn']   = linear_trans_s12
params_synthetic['synthetic12']['obs_fxn']     = linear_obs_s12
params_synthetic['synthetic12']['init_mu']     = np.ones((10,))*0.

params_synthetic['synthetic12']['init_cov']    = 0.1
params_synthetic['synthetic12']['init_cov_full']    = np.zeros((10,10))
np.fill_diagonal(params_synthetic['synthetic12']['init_cov_full'],0.1)

params_synthetic['synthetic12']['baseline']    = 'KF' 
params_synthetic['synthetic12']['docstr']      = '$z_t\sim\mathcal{N}(W_tz_{t-1}+b_t,10)$\n$x_t\sim\mathcal{N}(W_oz_t,20)$'



params_synthetic['synthetic13'] = {}
params_synthetic['synthetic13']['dim_obs']     = 200
params_synthetic['synthetic13']['dim_stoc']    = 100
params_synthetic['synthetic13']['params']      = {}

params_synthetic['synthetic13']['trans_cov']   = 10.
params_synthetic['synthetic13']['trans_cov_full']   = np.zeros((100,100))
np.fill_diagonal(params_synthetic['synthetic13']['trans_cov_full'],10.)

params_synthetic['synthetic13']['trans_drift'] = saved_matrices['btrans_100'] 
params_synthetic['synthetic13']['trans_mult']  = saved_matrices['Wtrans_100']

params_synthetic['synthetic13']['obs_cov']     = 20.
params_synthetic['synthetic13']['obs_cov_full']= np.zeros((200,200))
np.fill_diagonal(params_synthetic['synthetic13']['obs_cov_full'],20.)

params_synthetic['synthetic13']['obs_drift']   = np.ones((200,))*0. 
params_synthetic['synthetic13']['obs_mult']    = saved_matrices['Wobs_100']
def linear_trans_s13(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices['Wtrans_100'])+saved_matrices['btrans_100']
def linear_obs_s13(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices['Wobs_100'])
params_synthetic['synthetic13']['trans_fxn']   = linear_trans_s13
params_synthetic['synthetic13']['obs_fxn']     = linear_obs_s13
params_synthetic['synthetic13']['init_mu']     = np.ones((100,))*0.

params_synthetic['synthetic13']['init_cov']    = 0.1
params_synthetic['synthetic13']['init_cov_full']  = np.zeros((100,100))
np.fill_diagonal(params_synthetic['synthetic13']['init_cov_full'],0.1)

params_synthetic['synthetic13']['baseline']    = 'KF' 
params_synthetic['synthetic13']['docstr']      = '$z_t\sim\mathcal{N}(W_tz_{t-1}+b_t,10)$\n$x_t\sim\mathcal{N}(W_oz_t,20)$'


params_synthetic['synthetic14'] = {}
params_synthetic['synthetic14']['dim_obs']     = 500
params_synthetic['synthetic14']['dim_stoc']    = 250
params_synthetic['synthetic14']['params']      = {}

params_synthetic['synthetic14']['trans_cov']   = 10.
params_synthetic['synthetic14']['trans_cov_full']   = np.zeros((250,250))
np.fill_diagonal(params_synthetic['synthetic14']['trans_cov_full'],10.)

params_synthetic['synthetic14']['trans_drift'] = saved_matrices['btrans_250'] 
params_synthetic['synthetic14']['trans_mult']  = saved_matrices['Wtrans_250']

params_synthetic['synthetic14']['obs_cov']     = 20.
params_synthetic['synthetic14']['obs_cov_full']     = np.zeros((500,500))
np.fill_diagonal(params_synthetic['synthetic14']['obs_cov_full'],20.)

params_synthetic['synthetic14']['obs_drift']   = np.ones((500,))*0. 
params_synthetic['synthetic14']['obs_mult']    = saved_matrices['Wobs_250']
def linear_trans_s14(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices['Wtrans_250'])+saved_matrices['btrans_250']
def linear_obs_s14(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices['Wobs_250'])
params_synthetic['synthetic14']['trans_fxn']   = linear_trans_s14
params_synthetic['synthetic14']['obs_fxn']     = linear_obs_s14
params_synthetic['synthetic14']['init_mu']     = np.ones((250,))*0.

params_synthetic['synthetic14']['init_cov']    = 0.1
params_synthetic['synthetic14']['init_cov_full']    = np.zeros((250,250))
np.fill_diagonal(params_synthetic['synthetic14']['init_cov_full'],0.1)

params_synthetic['synthetic14']['baseline']    = 'KF' 
params_synthetic['synthetic14']['docstr']      = '$z_t\sim\mathcal{N}(W_tz_{t-1}+b_t,10)$\n$x_t\sim\mathcal{N}(W_oz_t,20)$'

"""
Dimobs and Dimz are the same
"""
params_synthetic['synthetic15'] = {}
params_synthetic['synthetic15']['dim_obs']     = 10
params_synthetic['synthetic15']['dim_stoc']    = 10
params_synthetic['synthetic15']['params']      = {}

params_synthetic['synthetic15']['trans_cov']   = 1.
params_synthetic['synthetic15']['trans_cov_full']   = np.zeros((10,10))
np.fill_diagonal(params_synthetic['synthetic15']['trans_cov_full'], 1.)

params_synthetic['synthetic15']['trans_drift'] = saved_matrices_2['btrans_10'] 
params_synthetic['synthetic15']['trans_mult']  = saved_matrices_2['Wtrans_10']

params_synthetic['synthetic15']['obs_cov']     = 10.
params_synthetic['synthetic15']['obs_cov_full']= np.zeros((10,10))
np.fill_diagonal(params_synthetic['synthetic15']['obs_cov_full'], 10.)

params_synthetic['synthetic15']['obs_drift']   = np.ones((10,))*0. 
params_synthetic['synthetic15']['obs_mult']    = saved_matrices_2['Wobs_10'] 
def linear_trans_s15(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wtrans_10'])+saved_matrices_2['btrans_10']
def linear_obs_s15(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wobs_10'])
params_synthetic['synthetic15']['trans_fxn']   = linear_trans_s15
params_synthetic['synthetic15']['obs_fxn']     = linear_obs_s15
params_synthetic['synthetic15']['init_mu']     = np.ones((10,))*0.

params_synthetic['synthetic15']['init_cov']    = 0.1
params_synthetic['synthetic15']['init_cov_full']    = np.zeros((10,10))
np.fill_diagonal(params_synthetic['synthetic15']['init_cov_full'],0.1)

params_synthetic['synthetic15']['baseline']    = 'KF' 
params_synthetic['synthetic15']['docstr']      = '$z_t\sim\mathcal{N}(W_tz_{t-1}+b_t,10)$\n$x_t\sim\mathcal{N}(W_oz_t,20)$'


params_synthetic['synthetic16'] = {}
params_synthetic['synthetic16']['dim_obs']     = 100
params_synthetic['synthetic16']['dim_stoc']    = 100
params_synthetic['synthetic16']['params']      = {}

params_synthetic['synthetic16']['trans_cov']   = 1.
params_synthetic['synthetic16']['trans_cov_full']   = np.zeros((100,100))
np.fill_diagonal(params_synthetic['synthetic16']['trans_cov_full'],1.)

params_synthetic['synthetic16']['trans_drift'] = saved_matrices_2['btrans_100'] 
params_synthetic['synthetic16']['trans_mult']  = saved_matrices_2['Wtrans_100']

params_synthetic['synthetic16']['obs_cov']     = 10.
params_synthetic['synthetic16']['obs_cov_full']= np.zeros((100,100))
np.fill_diagonal(params_synthetic['synthetic16']['obs_cov_full'],10.)

params_synthetic['synthetic16']['obs_drift']   = np.ones((100,))*0. 
params_synthetic['synthetic16']['obs_mult']    = saved_matrices_2['Wobs_100']
def linear_trans_s16(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wtrans_100'])+saved_matrices_2['btrans_100']
def linear_obs_s16(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wobs_100'])
params_synthetic['synthetic16']['trans_fxn']   = linear_trans_s16
params_synthetic['synthetic16']['obs_fxn']     = linear_obs_s16
params_synthetic['synthetic16']['init_mu']     = np.ones((100,))*0.

params_synthetic['synthetic16']['init_cov']    = 0.1
params_synthetic['synthetic16']['init_cov_full']  = np.zeros((100,100))
np.fill_diagonal(params_synthetic['synthetic16']['init_cov_full'],0.1)

params_synthetic['synthetic16']['baseline']    = 'KF' 
params_synthetic['synthetic16']['docstr']      = '$z_t\sim\mathcal{N}(W_tz_{t-1}+b_t,10)$\n$x_t\sim\mathcal{N}(W_oz_t,20)$'

params_synthetic['synthetic17'] = {}
params_synthetic['synthetic17']['dim_obs']     = 250 
params_synthetic['synthetic17']['dim_stoc']    = 250
params_synthetic['synthetic17']['params']      = {}

params_synthetic['synthetic17']['trans_cov']   = 1.
params_synthetic['synthetic17']['trans_cov_full']   = np.zeros((250,250))
np.fill_diagonal(params_synthetic['synthetic17']['trans_cov_full'],1.)

params_synthetic['synthetic17']['trans_drift'] = saved_matrices_2['btrans_250'] 
params_synthetic['synthetic17']['trans_mult']  = saved_matrices_2['Wtrans_250']

params_synthetic['synthetic17']['obs_cov']     = 10.
params_synthetic['synthetic17']['obs_cov_full']     = np.zeros((250,250))
np.fill_diagonal(params_synthetic['synthetic17']['obs_cov_full'],10.)

params_synthetic['synthetic17']['obs_drift']   = np.ones((250,))*0. 
params_synthetic['synthetic17']['obs_mult']    = saved_matrices_2['Wobs_250']
def linear_trans_s17(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wtrans_250'])+saved_matrices_2['btrans_250']
def linear_obs_s17(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wobs_250'])
params_synthetic['synthetic17']['trans_fxn']   = linear_trans_s17
params_synthetic['synthetic17']['obs_fxn']     = linear_obs_s17
params_synthetic['synthetic17']['init_mu']     = np.ones((250,))*0.
params_synthetic['synthetic17']['init_cov']    = 0.1
params_synthetic['synthetic17']['init_cov_full']    = np.zeros((250,250))
np.fill_diagonal(params_synthetic['synthetic17']['init_cov_full'],0.1)
params_synthetic['synthetic17']['baseline']    = 'KF' 
params_synthetic['synthetic17']['docstr']      = '$z_t\sim\mathcal{N}(W_tz_{t-1}+b_t,10)$\n$x_t\sim\mathcal{N}(W_oz_t,20)$'


"""
dimz and dimobs are the same with diagonal 
"""

params_synthetic['synthetic18'] = {}
params_synthetic['synthetic18']['dim_obs']     = 10
params_synthetic['synthetic18']['dim_stoc']    = 10
params_synthetic['synthetic18']['params']      = {}

params_synthetic['synthetic18']['trans_cov']   = 1.
params_synthetic['synthetic18']['trans_cov_full']   = np.zeros((10,10))
np.fill_diagonal(params_synthetic['synthetic18']['trans_cov_full'], 1.)

params_synthetic['synthetic18']['trans_drift'] = saved_matrices_2['btrans_10_diag'] 
params_synthetic['synthetic18']['trans_mult']  = saved_matrices_2['Wtrans_10_diag']

params_synthetic['synthetic18']['obs_cov']     = 10.
params_synthetic['synthetic18']['obs_cov_full']= np.zeros((10,10))
np.fill_diagonal(params_synthetic['synthetic18']['obs_cov_full'], 10.)

params_synthetic['synthetic18']['obs_drift']   = np.ones((10,))*0. 
params_synthetic['synthetic18']['obs_mult']    = saved_matrices_2['Wobs_10_diag'] 
def linear_trans_s18(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wtrans_10_diag'])+saved_matrices_2['btrans_10_diag']
def linear_obs_s18(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wobs_10_diag'])
params_synthetic['synthetic18']['trans_fxn']   = linear_trans_s18
params_synthetic['synthetic18']['obs_fxn']     = linear_obs_s18
params_synthetic['synthetic18']['init_mu']     = np.ones((10,))*0.

params_synthetic['synthetic18']['init_cov']    = 0.1
params_synthetic['synthetic18']['init_cov_full']    = np.zeros((10,10))
np.fill_diagonal(params_synthetic['synthetic18']['init_cov_full'],0.1)

params_synthetic['synthetic18']['baseline']    = 'KF' 
params_synthetic['synthetic18']['docstr']      = '$z_t\sim\mathcal{N}(W_tz_{t-1}+b_t,10)$\n$x_t\sim\mathcal{N}(W_oz_t,20)$'


params_synthetic['synthetic19'] = {}
params_synthetic['synthetic19']['dim_obs']     = 100
params_synthetic['synthetic19']['dim_stoc']    = 100
params_synthetic['synthetic19']['params']      = {}

params_synthetic['synthetic19']['trans_cov']   = 1.
params_synthetic['synthetic19']['trans_cov_full']   = np.zeros((100,100))
np.fill_diagonal(params_synthetic['synthetic19']['trans_cov_full'],1.)

params_synthetic['synthetic19']['trans_drift'] = saved_matrices_2['btrans_100_diag'] 
params_synthetic['synthetic19']['trans_mult']  = saved_matrices_2['Wtrans_100_diag']

params_synthetic['synthetic19']['obs_cov']     = 10.
params_synthetic['synthetic19']['obs_cov_full']= np.zeros((100,100))
np.fill_diagonal(params_synthetic['synthetic19']['obs_cov_full'],10.)

params_synthetic['synthetic19']['obs_drift']   = np.ones((100,))*0. 
params_synthetic['synthetic19']['obs_mult']    = saved_matrices_2['Wobs_100_diag']
def linear_trans_s19(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wtrans_100_diag'])+saved_matrices_2['btrans_100_diag']
def linear_obs_s19(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wobs_100_diag'])
params_synthetic['synthetic19']['trans_fxn']   = linear_trans_s19
params_synthetic['synthetic19']['obs_fxn']     = linear_obs_s19
params_synthetic['synthetic19']['init_mu']     = np.ones((100,))*0.

params_synthetic['synthetic19']['init_cov']    = 0.1
params_synthetic['synthetic19']['init_cov_full']  = np.zeros((100,100))
np.fill_diagonal(params_synthetic['synthetic19']['init_cov_full'],0.1)

params_synthetic['synthetic19']['baseline']    = 'KF' 
params_synthetic['synthetic19']['docstr']      = '$z_t\sim\mathcal{N}(W_tz_{t-1}+b_t,10)$\n$x_t\sim\mathcal{N}(W_oz_t,20)$'

params_synthetic['synthetic20'] = {}
params_synthetic['synthetic20']['dim_obs']     = 250 
params_synthetic['synthetic20']['dim_stoc']    = 250
params_synthetic['synthetic20']['params']      = {}

params_synthetic['synthetic20']['trans_cov']   = 1.
params_synthetic['synthetic20']['trans_cov_full']   = np.zeros((250,250))
np.fill_diagonal(params_synthetic['synthetic20']['trans_cov_full'],1.)

params_synthetic['synthetic20']['trans_drift'] = saved_matrices_2['btrans_250_diag'] 
params_synthetic['synthetic20']['trans_mult']  = saved_matrices_2['Wtrans_250_diag']

params_synthetic['synthetic20']['obs_cov']     = 10.
params_synthetic['synthetic20']['obs_cov_full']     = np.zeros((250,250))
np.fill_diagonal(params_synthetic['synthetic20']['obs_cov_full'],10.)

params_synthetic['synthetic20']['obs_drift']   = np.ones((250,))*0. 
params_synthetic['synthetic20']['obs_mult']    = saved_matrices_2['Wobs_250_diag']
def linear_trans_s20(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wtrans_250_diag'])+saved_matrices_2['btrans_250_diag']
def linear_obs_s20(z,fxn_params = {},ns=None): 
    return np.dot(z,saved_matrices_2['Wobs_250_diag'])
params_synthetic['synthetic20']['trans_fxn']   = linear_trans_s20
params_synthetic['synthetic20']['obs_fxn']     = linear_obs_s20
params_synthetic['synthetic20']['init_mu']     = np.ones((250,))*0.
params_synthetic['synthetic20']['init_cov']    = 0.1
params_synthetic['synthetic20']['init_cov_full']    = np.zeros((250,250))
np.fill_diagonal(params_synthetic['synthetic20']['init_cov_full'],0.1)
params_synthetic['synthetic20']['baseline']    = 'KF' 
params_synthetic['synthetic20']['docstr']      = '$z_t\sim\mathcal{N}(W_tz_{t-1}+b_t,10)$\n$x_t\sim\mathcal{N}(W_oz_t,20)$'
