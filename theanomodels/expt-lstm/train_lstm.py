import os,time
from ..datasets.load import loadDataset
from ..utils.parse_args_lstm import params 
from ..utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime

if params['dataset']=='':
    params['dataset']='jsb'
dataset = loadDataset(params['dataset'])
params['savedir']+='-'+params['dataset']
createIfAbsent(params['savedir'])

#Saving/loading
for k in ['dim_observations','data_type']:
    params[k] = dataset[k]
mapPrint('Options: ',params)

#Setup VAE Model (or reload from existing savefile)
start_time = time.time()
from models.lstm import LSTM
displayTime('import LSTM',start_time, time.time())
lstm    = None

#Remove from params
start_time = time.time()
removeIfExists('./NOSUCHFILE')
reloadFile = params.pop('reloadFile')
if os.path.exists(reloadFile):
    pfile=params.pop('paramFile')
    assert os.path.exists(pfile),pfile+' not found. Need paramfile'
    print 'Reloading trained model from : ',reloadFile
    print 'Assuming ',pfile,' corresponds to model'
    lstm  = LSTM(params, paramFile = pfile, reloadFile = reloadFile) 
else:
    pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'
    print 'Training model from scratch. Parameters in: ',pfile
    lstm  = LSTM(params, paramFile = pfile)
displayTime('Building lstm',start_time, time.time())


savef     = os.path.join(params['savedir'],params['unique_id']) 
print 'Savefile: ',savef
start_time= time.time()
savedata = lstm.learn(dataset['train'], dataset['mask_train'], 
                                epoch_start =0 , 
                                epoch_end = params['epochs'], 
                                batch_size = params['batch_size'],
                                savefreq   = params['savefreq'],
                                savefile   = savef,
                                dataset_eval=dataset['valid'],
                                mask_eval  = dataset['mask_valid']
                                )
displayTime('Running LSTM',start_time, time.time())
savedata['test_nll'] = lstm.evaluateNLL(dataset['test'], dataset['mask_test'], batch_size = params['batch_size'])
#Save file log file
saveHDF5(savef+'-final.h5',savedata)

print 'TEST LL: ',savedata['test_nll']
import ipdb;ipdb.set_trace()
