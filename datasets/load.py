import os,time,h5py,process
import numpy as np
from utils.misc import getPYDIR

def loadDataset(dataset, **kwargs):
    """
    loadDataset(dataset)
    dataset: Name of dataset
    returns: Map with properties of dataset
    """
    if dataset=='binarized_mnist':
        return _binarized_mnist()
    if dataset=='mnist':
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
    else:
        assert False,'invalid dataset: '+dataset


def _mnist():
    """
    Utility function to process & load MNIST dataset
    """
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
    """
    Utility functino to process and load synthetic datasets
    """
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
    datasets['dim_actions']      = 0
    datasets['hasMasks']         = False
    datasets['data_type']        = 'real'
    ff.close()
    return datasets
