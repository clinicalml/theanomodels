import os,time
import numpy as np
from datasets.load import loadDataset
from utils.parse_args_vae import params 
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime

dataset = 'binarized_mnist'
params['savedir']+='-'+dataset
createIfAbsent(params['savedir'])
dataset = loadDataset(dataset)

#Saving/loading
for k in ['dim_observations','data_type']:
    params[k] = dataset[k]
mapPrint('Options: ',params)

#Setup VAE Model (or reload from existing savefile)
start_time = time.time()
from models.vae import VAE
displayTime('import vae',start_time, time.time())
vae    = None

#Remove from params
start_time = time.time()
removeIfExists('./NOSUCHFILE')
reloadFile = params.pop('reloadFile')
if os.path.exists(reloadFile):
    pfile=params.pop('paramFile')
    assert os.path.exists(pfile),pfile+' not found. Need paramfile'
    print 'Reloading trained model from : ',reloadFile
    print 'Assuming ',pfile,' corresponds to model'
    vae  = VAE(params, paramFile = pfile, reloadFile = reloadFile) 
else:
    pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'
    print 'Training model from scratch. Parameters in: ',pfile
    vae  = VAE(params, paramFile = pfile)
displayTime('Building vae',start_time, time.time())

savef      = os.path.join(params['savedir'],params['unique_id']) 

start_time = time.time()

replicate_K = 5

#trainData = np.concatenate([dataset['train'],dataset['valid']],axis=0);validData = dataset['test']
trainData = dataset['train'];validData = dataset['valid']
savedata = vae.learn(           trainData,
                                epoch_start=0 , 
                                epoch_end  = params['epochs'], 
                                batch_size = params['batch_size'],
                                savefreq   = params['savefreq'],
                                savefile   = savef,
                                dataset_eval= validData,
                                replicate_K= replicate_K
                                )
displayTime('Running VAE',start_time, time.time())
savedata['test_bound'] = vae.evaluateBound(dataset['test'], batch_size = params['batch_size'])

#Save file log file
saveHDF5(savef+'-final.h5',savedata)
print 'Test Bound: ',savedata['test_bound']

import ipdb;
ipdb.set_trace()
