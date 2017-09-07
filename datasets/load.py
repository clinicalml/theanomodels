import os,time,h5py,process
import numpy as np
from utils.misc import getPYDIR, readPickle
import urllib,tarfile

def loadDataset(dataset, **kwargs):
    """
    loadDataset(dataset)
    dataset: Name of dataset
    returns: Map with properties of dataset
    """
    if dataset=='binarized_mnist':
        return _binarized_mnist()
    elif dataset=='fashion_mnist':
        return _mnist(fashion=True)
    elif dataset=='mnist':
        return _mnist()
    elif dataset in ['jsb','nottingham', 'piano', 'musedata', 'jsb-sorted','nottingham-sorted', 'piano-sorted', 'musedata-sorted']:
        return _polyphonic(dataset)
    elif 'synthetic' in dataset:
        return _synthetic(dataset)
    elif 'months' in dataset:
        return _medical(dataset, **kwargs)
    elif 'MOCAP' in dataset:
        return _MOCAP()
    elif dataset in ['iamondb','blizzard','accent']:
        return _iamondb_or_speech(dataset, **kwargs)
    elif dataset=='cifar10':
        return _cifar10()
    else:
        assert False,'invalid dataset: '+dataset

def reshapeMatrix(mat, w = 32, h = 32):
    #input bs x (32*32)*3 
    prod = w*h
    ch1  = mat[:,:prod].reshape(-1,1,h,w)
    ch2  = mat[:,prod:2*prod].reshape(-1,1,h,w)
    ch3  = mat[:,2*prod:].reshape(-1,1,h,w)
    return np.concatenate([ch1,ch2,ch3],axis=1)

def _cifar10():
    #CIFAR 10 Dataset
    DIR       = getPYDIR()+'/datasets/cifar10'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    savef = os.path.join(DIR,'cifar-10-python.tar.gz')
    if not os.path.exists(savef):
        urllib.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', savef)
    cifarfile = os.path.join(DIR,'cifar10.h5') 
    if not os.path.exists(cifarfile):
        print 'Extracting CIFAR...'
        tf    = tarfile.open(savef)
        tf.extractall(DIR)
        tf.close()
        EDIR  = DIR+'/cifar-10-batches-py/'
        h5f   = h5py.File(cifarfile,mode='w')
        traindatalist,trainlabellist=[],[]
        for k in range(5):
            print k,
            hmap = readPickle(EDIR+'/data_batch_'+str(k+1))[0]
            traindatalist.append(hmap['data'])
            trainlabellist.append(hmap['labels'])
        alltrainx= np.concatenate(traindatalist,axis=0)
        alltrainy= np.concatenate(trainlabellist,axis=0)
        np.random.seed(1)
        idxlist  = np.random.permutation(alltrainx.shape[0])
        val_idx  = idxlist[:int(0.1*alltrainx.shape[0])]
        tr_idx   = idxlist[int(0.1*alltrainx.shape[0]):]
        TRAINX   = alltrainx[tr_idx]
        TRAINY   = alltrainy[tr_idx]
        VALIDX   = alltrainx[val_idx]
        VALIDY   = alltrainy[val_idx]
        h5f.create_dataset('train'  , data=reshapeMatrix(TRAINX))
        h5f.create_dataset('valid'  , data=reshapeMatrix(VALIDX))
        h5f.create_dataset('train_y', data=TRAINY)
        h5f.create_dataset('valid_y', data=VALIDY)
        hmap     = readPickle(EDIR+'/test_batch')[0]
        h5f.create_dataset('test', data=reshapeMatrix(hmap['data']))
        h5f.create_dataset('test_y', data=np.array(hmap['labels']))
        hmap     = readPickle(EDIR+'/batches.meta')[0]
        h5f.create_dataset('label_names', data=np.array(hmap['label_names'],dtype='|S10'))
        h5f.close()
        print '\nCreated CIFAR h5 file'
    else:
        print 'Found CIFAR h5 file'
    h5f       = h5py.File(cifarfile,mode='r')
    dataset   = {}
    dataset['label_names'] = h5f['label_names'].value
    dataset['train'] = h5f['train'].value
    dataset['test']  = h5f['test'].value
    dataset['valid'] = h5f['valid'].value
    dataset['train_y']=h5f['train_y'].value
    dataset['test_y'] =h5f['test_y'].value
    dataset['valid_y']=h5f['valid_y'].value
    dataset['dim_observations'] = np.prod(dataset['train'].shape[1:])
    dataset['num_channels'] = dataset['train'].shape[-3] 
    dataset['dim_h']  = dataset['train'].shape[-2] 
    dataset['dim_w']  = dataset['train'].shape[-1] 
    dataset['data_type']  = 'image'
    h5f.close()
    return dataset

def _mnist(fashion=False):
    """
    Utility function to process & load MNIST dataset
    """
    if fashion:
        pfile = process._processFashionMNIST()
    else:
        pfile = process._processMNIST()
    ff = h5py.File(pfile,mode='r')
    datasets = {}
    datasets['train']     = ff['train'].value
    datasets['test']      = ff['test'].value
    datasets['valid']     = ff['valid'].value
    datasets['train_y']     = ff['train_y'].value
    datasets['test_y']      = ff['test_y'].value
    datasets['valid_y']     = ff['valid_y'].value
    datasets['dim_observations'] = datasets['train'].shape[1]
    datasets['hasMasks']  = False
    datasets['data_type'] = 'binary'
    ff.close()
    return datasets 

def _binarized_mnist():
    """
    Utility function to process & load MNIST dataset
    """
    pfile = process._processBinarizedMNIST()
    ff = h5py.File(pfile,mode='r')
    datasets = {}
    datasets['train']     = ff['train'].value
    datasets['test']      = ff['test'].value
    datasets['valid']     = ff['valid'].value
    datasets['dim_observations'] = datasets['train'].shape[1]
    datasets['hasMasks']  = False
    datasets['data_type'] = 'binary'
    ff.close()
    return datasets 

def _polyphonic(dset):
    """
    Utility function to process & load polyphonic datasets 
    """
    pfile = process._processPolyphonic(dset)
    ff = h5py.File(pfile,mode='r')
    datasets = {}
    datasets['train']            = ff['train'].value
    datasets['test']             = ff['test'].value
    datasets['valid']            = ff['valid'].value
    datasets['mask_train']       = ff['mask_train'].value
    datasets['mask_test']        = ff['mask_test'].value
    datasets['mask_valid']       = ff['mask_valid'].value
    datasets['dim_observations'] = datasets['train'].shape[2]
    datasets['dim_actions']      = 0
    datasets['hasMasks']         = True
    datasets['data_type']        = 'binary'
    ff.close()
    return datasets

def _synthetic(dset):
    """ Utility functino to process and load synthetic datasets """
    pfile = process._processSynthetic(dset)
    ff = h5py.File(pfile,mode='r')
    datasets = {}
    datasets['train']            = ff['train'].value
    datasets['test']             = ff['test'].value
    datasets['valid']            = ff['valid'].value
    datasets['mask_train']       = np.ones((datasets['train'].shape[0], datasets['train'].shape[1]))
    datasets['mask_test']        = np.ones((datasets['test'].shape[0],  datasets['test'].shape[1] ))
    datasets['mask_valid']       = np.ones((datasets['valid'].shape[0], datasets['valid'].shape[1]))
    datasets['train_z']          = ff['train_z'].value
    datasets['test_z']           = ff['test_z'].value
    datasets['valid_z']          = ff['valid_z'].value
    datasets['dim_observations'] = datasets['train'].shape[2]
    datasets['dim_stochastic'] = datasets['train_z'].shape[2]
    datasets['dim_actions']      = 0
    datasets['hasMasks']         = False
    datasets['data_type']        = 'real'
    ff.close()
    return datasets

if __name__=='__main__':
    #dset = loadDataset('cifar10')
    dset = loadDataset('mnist')
    dsetf = loadDataset('fashion_mnist')
    import ipdb;ipdb.set_trace()
