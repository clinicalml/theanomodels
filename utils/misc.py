import os,h5py,sys,glob
import numpy as np
import cPickle as pickle

def readPickle(fname, nobjects =1):
    obj = []
    with open(fname,'rb') as f:
        for n in range(nobjects):
            obj.append(pickle.load(f))
    return obj

def savePickle(list_of_objects, fname):
    with open(fname,'wb') as f:
        for obj in list_of_objects:
            pickle.dump(obj, f)

def removeIfExists(fname):
    if os.path.exists(fname):
        os.remove(fname)

def createIfAbsent(DIR):
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)

def mapPrint(title, hmap, nlFreq=1):
    print title,'   {'
    ctr = 0
    for k in hmap:
        print k,':',hmap[k], 
        ctr+=1
        if ctr%nlFreq==0:
            print '\n',
    print '\n}'

def saveHDF5(fname, savedata):
    ff = h5py.File(fname,mode='w')
    for k in savedata:
        ff.create_dataset(k, data=savedata[k])
    ff.close()

def displayTime(event, start, end, reportingIn='seconds'):
    time_taken = end-start
    if reportingIn=='minutes':
        time_taken= time_taken/60.
    if reportingIn=='hours':
        time_taken= time_taken/3600.
    print '\t\t<',event,'> took ',time_taken,' ',reportingIn

def getPYDIR():
    for k in sys.path:
        if 'theanomodels' in os.path.basename(k):
            return k.split('theanomodels')[0]+'theanomodels'
    assert False,'Should not reach here, directory <theanomodels> expected in PYTHONPATH.'

def loadHDF5(fname):
    assert os.path.exists(fname),'File not found'
    fhandle = h5py.File(fname,mode='r')
    results = {}
    for k in fhandle.keys():
        results[k] = fhandle[k].value
    fhandle.close()
    return results

def getLowestError(mat):
    """ 
    Get the lowest error in Kx2 matrix. Col 0: Epochs, Col 1: Val. Error
    """
    idxMin  = np.argmin(mat[:,1])
    epochMin= mat[int(idxMin),0]
    valMin  = mat[int(idxMin),1]
    return epochMin, valMin, idxMin

def setNumpyFloatPrint():
    """
    Set numpy's print so you can see numbers while debugging
    """
    np.set_printoptions(formatter={'float':lambda x: '%.4f'%(x)})

def getBestStatistics(fstring):
    """
    Get the stats file with the best validation error
    -Get the last number
    """
    maxEpoch = 0
    for f in glob.glob(fstring+'*-stats.h5'):
        if 'EP' in f:
            epoch = int(f.split('-EP')[1].split('-')[0])
            if epoch > maxEpoch and epoch>0:
                maxEpoch = epoch
                maxF     = f
    if maxEpoch==0:
        return {}
    data = loadHDF5(maxF)
    epochMin, valMin, idxMin = getLowestError(data['valid_bound'])
    results = {}
    results['maxEpoch'] = maxEpoch
    results['maxF']     = maxF
    results['epochMin'] = epochMin
    results['valMin']   = valMin
    results['idxMin']   = idxMin
    results['minF']   =   maxF.replace(str(int(maxEpoch)),str(int(epochMin)))
    print maxEpoch, maxF, epochMin, valMin, idxMin, results['minF'],'\n'
    return results

def trainValidTestIdx(N, train_frac = 0.80, test_frac  = 0.5):
    np.random.seed(1)
    idxlist = np.random.permutation(N)
    Ntrain  = int(train_frac*N)
    train_idx, test_valid_idx = idxlist[:Ntrain], idxlist[Ntrain:]
    Ntestvalid= len(test_valid_idx)
    Ntest     = int(test_frac*Ntestvalid)
    valid_idx = test_valid_idx[:Ntest]
    test_idx  = test_valid_idx[Ntest:]
    assert len(train_idx)+len(valid_idx)+len(test_idx)==N,'Index lengths dont match'
    return train_idx, valid_idx, test_idx

def getConfigFile(fname):
    return fname.replace('final.h5','').split('EP')[0]+'config.pkl'

def getUniqueIDFromParams(l_params, short_names = {}):
    """ 
        Inputs: l_params is a list of parameter hashmaps
        Assumption: l_params[0] is a hashmap that contains keys found in l_params[idx] for all idx 
        Goal: Return strings corresponding to keys k such that l_params[idx][k]~=l_params[!idx][k] 
        useful when looking at runs across different configurations
        short_names is an optional map that could contain short forms for some/all of the keys in params
                    if unavailable, the long name is used
    """ 
    all_keys = l_params[0].keys()
    umap     = {}
    names    = ['' for p in l_params]
    for k in all_keys:
        if k in ['unique_id','savedir']:
            continue
        unique_vals = set([l_params[idx][k] for idx in range(len(l_params))])
        kname = k
        if k in short_names:
            kname = short_names[k]
        if len(unique_vals)>1:
            for idx,p in enumerate(l_params):
                names[idx]+='-'+kname+'-'+str(p[k])
    return names

def productOfBernoullisMLE(train, test):
    """
        MLE estimator for product of bernoullis
        Input: train/test datasets (2D numpy arrays)
        Output:value of NLL, ML params
    """
    assert train.ndim==2 and test.ndim==2,'Expecting 2D arrays for train/test'
    if train.__class__.__name__=='ndarray':
        assert np.unique(train).sum() in [0.,1.] and np.unique(test).sum() in [0.,1.],'Expecting sum to be 0/1'
    elif train.__class__.__name__=='csr_matrix' or train.__class__.__name__=='csc_matrix':
        assert sum(set(train.data))==1. in [0.,1.] and sum(set(test.data)) in [0.,1.],'Expecting sum to be in 0/1'
    else:
        assert False,'Invalid input type: '+train.__class__.__name__
    eps = 1e-6
    if train.__class__.__name__=='ndarray':
        params      = train.mean(0,keepdims=True)
        NLL_test    = -(np.log(params+eps)*test + np.log(1.-params+eps)*(1.-test))
        NLL_train   = -(np.log(params+eps)*train+ np.log(1.-params+eps)*(1.-train))
        return NLL_train.sum()/float(train.shape[0]), NLL_test.sum()/float(test.shape[0]), params
    else: #must be sparse
        params      = np.asarray(train.mean(0))
        if train.__class__.__name__=='csc_matrix' or test.__class__.__name__=='csc_matrix':
            train_mat = train.tocsr()
            test_mat  = test.tocsr()
        else:
            train_mat = train
            test_mat  = test
        NLL_train, NLL_test = 0., 0.
        for k in range(max(train_mat.shape[0],test.shape[0])):
            if k<train.shape[0]:
                NLL_train += (-(np.log(params+eps)*train_mat[k].toarray()+ np.log(1.-params+eps)*(1.-train_mat[k].toarray()))).sum()
            if k<test.shape[0]:
                NLL_test  += (-(np.log(params+eps)*test_mat[k].toarray() + np.log(1.-params+eps)*(1.-test_mat[k].toarray()))).sum() 
        NLL_test  /=float(test.shape[0])
        NLL_train /=float(train.shape[0])
        return NLL_train, NLL_test, params

if __name__=='__main__':
    pass
