import os,time
import numpy as np
from theanomodels.datasets.load import loadDataset
from theanomodels.utils.parse_args_mlp import params 
from theanomodels.utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime

dataset = 'mnist'
params['savedir']+='-'+dataset
createIfAbsent(params['savedir'])
dataset = loadDataset(dataset)

#Saving/loading
for k in ['dim_observations','data_type']:
    params[k] = dataset[k]
mapPrint('Options: ',params)

#Setup MLP Model (or reload from existing savefile)
start_time = time.time()
from theanomodels.models.mlp import MLP
displayTime('import MLP',start_time, time.time())
model    = None

#Remove from params
start_time = time.time()
removeIfExists('./NOSUCHFILE')
reloadFile = params.pop('reloadFile')
if os.path.exists(reloadFile):
    pfile=params.pop('paramFile')
    assert os.path.exists(pfile),pfile+' not found. Need paramfile'
    print 'Reloading trained model from : ',reloadFile
    print 'Assuming ',pfile,' corresponds to model'
    model  = MLP(params, paramFile = pfile, reloadFile = reloadFile) 
else:
    pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'
    print 'Training model from scratch. Parameters in: ',pfile
    model  = MLP(params, paramFile = pfile)
displayTime('Building MLP',start_time, time.time())

savef      = os.path.join(params['savedir'],params['unique_id'])+'-' 

start_time = time.time()

#trainData = {'X':dataset['train'], 'Y':dataset['train_y']}
#validData = {'X':dataset['valid'], 'Y':dataset['valid_y']}
#testData = {'X':dataset['test'], 'Y':dataset['test_y']}

trainData = {'X':np.vstack((dataset['train'],dataset['valid'])),
             'Y':np.hstack((dataset['train_y'],dataset['valid_y']))}
validData = {'X':dataset['test'], 'Y':dataset['test_y']}
testData = {'X':dataset['test'], 'Y':dataset['test_y']}

savedata = model.learn(         trainData,
                                epoch_start=0 , 
                                epoch_end  = params['epochs'], 
                                batch_size = params['batch_size'],
                                savefreq   = params['savefreq'],
                                evalfreq   = params['evalfreq'],
                                savefile   = savef,
                                dataset_eval= validData,
                                )
displayTime('Running MLP',start_time, time.time())
savedata['test_crossentropy'], savedata['test_accuracy'] = model.evaluateClassifier(testData, batch_size = params['batch_size'])

#Save file log file
saveHDF5(savef+'-final.h5',savedata)
print 'Test Bound: ',savedata['test_crossentropy']
print 'Test Accuracy: ',savedata['test_accuracy']

#import ipdb;
#ipdb.set_trace()
