#Scripts to process datasets
import h5py,os,urllib,cPickle,gzip
import numpy as np
from synthp import params_synthetic
from utils.misc import getPYDIR

def _processMNIST():
    pfile = getPYDIR()+'/datasets/mnist/proc-mnist.h5'
    """
        Move to processed h5 file
    """
    DIR = os.path.dirname(pfile)
    if not os.path.exists(DIR):
        print 'Making: ',DIR
        os.mkdir(DIR)
    if not os.path.exists(os.path.join(DIR,'mnist.pkl.gz')):
        print 'Downloading data'
        urllib.urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz',os.path.join(DIR,'mnist.pkl.gz'))
    if os.path.exists(pfile):
        print 'Found: ',pfile
        return pfile
    print 'Processing MNIST'
    f = gzip.open(os.path.join(DIR,'mnist.pkl.gz'))
    train, valid, test = cPickle.load(f)
    f.close()
    h5f   = h5py.File(pfile, mode='w')
    h5f.create_dataset('train',data = train[0])
    h5f.create_dataset('train_y',data = train[1])
    h5f.create_dataset('test' ,data = test[0])
    h5f.create_dataset('test_y' ,data = test[1])
    h5f.create_dataset('valid',data = valid[0])
    h5f.create_dataset('valid_y',data = valid[1])
    h5f.close()
    print 'Done processing MNIST'
    return pfile
def _processBinarizedMNIST():
    pfile = getPYDIR()+'/datasets/mnist/proc-bmnist.h5'
    """
        Move to processed h5 file
    """
    DIR = os.path.dirname(pfile)
    if not os.path.exists(DIR):
        print 'Making: ',DIR
        os.mkdir(DIR)
    if not os.path.exists(os.path.join(DIR,'binarized_mnist_train.amat')):
        print 'Downloading binarized mnist'
        urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat',os.path.join(DIR,'binarized_mnist_train.amat'))
        urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat',os.path.join(DIR,'binarized_mnist_valid.amat'))
        urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat',os.path.join(DIR,'binarized_mnist_test.amat'))
    if os.path.exists(pfile):
        print 'Found: ',pfile
        return pfile
    print 'Processing binarized MNIST'
    h5f   = h5py.File(pfile, mode='w')
    h5f.create_dataset('train',data = np.loadtxt(os.path.join(DIR,'binarized_mnist_train.amat')))
    h5f.create_dataset('test' ,data = np.loadtxt(os.path.join(DIR,'binarized_mnist_test.amat')))
    h5f.create_dataset('valid',data = np.loadtxt(os.path.join(DIR,'binarized_mnist_valid.amat')))
    h5f.close()
    print 'Done processing binarized MNIST'
    return pfile

def _processPolyphonic(name):
    DIR = getPYDIR()+'/datasets'
    assert os.path.exists(DIR),'Directory does not exist: '+DIR
    polyphonicDIR = DIR+'/polyphonic/'
    if not os.path.exists(polyphonicDIR):
        os.mkdir(polyphonicDIR)
    fname = polyphonicDIR+'/'+name+'.h5'
    if os.path.exists(fname):
        print 'Found: ',fname
        return fname
    #Setup polyphonic datasets from scratch
    if not os.path.exists(os.path.join(polyphonicDIR,'piano.pkl')) or \
    not os.path.exists(os.path.join(polyphonicDIR,'musedata.pkl')) or \
    not os.path.exists(os.path.join(polyphonicDIR,'jsb.pkl')) or \
    not os.path.exists(os.path.join(polyphonicDIR,'nottingham.pkl')):
        print 'Downloading polyphonic pickled data into: ',polyphonicDIR
        os.system('wget '+'http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle -O '+os.path.join(polyphonicDIR,'jsb.pkl'))
        os.system('wget '+'http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle -O '+os.path.join(polyphonicDIR,'nottingham.pkl'))
        os.system('wget '+'http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle -O '+os.path.join(polyphonicDIR,'musedata.pkl'))
        os.system('wget '+'http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle -O '+os.path.join(polyphonicDIR,'piano.pkl'))
    else:
        print 'Polyphonic pickle files found'
    #Helper function to sort by sequence length
    def getSortedVersion(data,mask):
        idx         = np.argsort(mask.sum(1))
        return data[idx,:,:], mask[idx,:]
    for dset in ['jsb','piano','nottingham','musedata','jsb-sorted','piano-sorted','nottingham-sorted','musedata-sorted']:
        datafile = os.path.join(polyphonicDIR,dset.replace('-sorted','')+'.pkl')
        savefile = os.path.join(polyphonicDIR,dset+'.h5')
        print '\n\nDataset: ',dset
        print ('Reading from: ',datafile)
        print ('Saving to:',savefile)
        MAX = 108
        MIN = 21
        DIM = MAX-MIN+1
        alldata = cPickle.load(file(datafile))
        h5file = h5py.File(savefile,mode='w')
        for dtype in ['train','valid','test']:
            print '----',dtype,'----'
            dataset = alldata[dtype]
            N_SEQUENCES = len(dataset)
            #First, find out the maximum number of sequences
            MAX_LENGTH  = max(len(dataset[k]) for k in range(N_SEQUENCES))
            print N_SEQUENCES,' sequences with max length ',MAX_LENGTH
            mask         = np.zeros((N_SEQUENCES, MAX_LENGTH))
            compileddata = np.zeros((N_SEQUENCES, DIM, MAX_LENGTH))
            for idxseq,seq in enumerate(dataset):
                T = len(seq)
                mask[idxseq,:T] = 1
                for t in range(T):
                    compileddata[idxseq,np.array(seq[t]).astype(int)-MIN,t]=1
            if 'sorted' in dset:
                compileddata,mask = getSortedVersion(compileddata,mask)
            #Save as bs x T x dim
            compileddata      = compileddata.swapaxes(1,2)
            print 'First and last lenghts: ',mask.sum(1)[1:5].tolist(),'....',mask.sum(1)[-5:].tolist()
            print 'Saving tensor data: ',compileddata.shape,' Mask: ',mask.shape
            h5file.create_dataset(dtype,data = compileddata)
            h5file.create_dataset('mask_'+dtype,data = mask)
        h5file.close()

def _processSynthetic(dset):
    DIR = getPYDIR()+'/datasets'
    assert os.path.exists(DIR),'Directory does not exist: '+DIR
    syntheticDIR = DIR+'/synthetic/'
    if not os.path.exists(syntheticDIR):
        os.mkdir(syntheticDIR)
    fname = syntheticDIR+'/'+dset+'.h5'
    if os.path.exists(fname):
        print 'Found: ',fname
        return fname
    #Setup polyphonic datasets from scratch
    np.random.seed(1)
    def sampleGaussian(mu, cov):
        assert type(cov) is float or type(cov) is np.array,'invalid type: '+str(cov)+' type: '+str(type(cov))
        return mu + np.random.randn(*mu.shape)*np.sqrt(cov)
    def createDataset(N, T, t_fxn, e_fxn, init_mu, init_cov, trans_cov, obs_cov):
        all_z = []
        z_prev= sampleGaussian(np.ones((N,1,1))*init_mu, init_cov)
        all_z.append(np.copy(z_prev))
        for t in range(T-1):
            z_prev = sampleGaussian(t_fxn(z_prev), trans_cov) 
            all_z.append(z_prev)
        Z_true= np.concatenate(all_z, axis=1)
        assert Z_true.shape[1]==T,'Expecting T in dim 2 of Z_true'
        X     = sampleGaussian(e_fxn(Z_true), obs_cov)
        return Z_true, X
    if not np.all([os.path.exists(os.path.join(syntheticDIR,fname+'.h5')) for fname in ['synthetic'+str(i) for i in range(1,10)]]):
        #Create all datasets
        for s in range(9,11):
            print 'Creating: ',s
            dataset = {}
            transition_fxn = params_synthetic['synthetic'+str(s)]['trans_fxn']
            emission_fxn   = params_synthetic['synthetic'+str(s)]['obs_fxn'] 
            init_mu        = params_synthetic['synthetic'+str(s)]['init_mu']
            init_cov       = params_synthetic['synthetic'+str(s)]['init_cov']
            trans_cov      = params_synthetic['synthetic'+str(s)]['trans_cov']
            obs_cov        = params_synthetic['synthetic'+str(s)]['obs_cov']
            Ntrain = 5000
            Ttrain = 25
            Ttest  = 50
            Nvalid = 500
            Ntest  = 500
            train_Z, train_dataset  = createDataset(Ntrain, Ttrain, transition_fxn, emission_fxn, init_mu, init_cov, trans_cov, obs_cov) 
            valid_Z, valid_dataset  = createDataset(Nvalid, Ttrain, transition_fxn, emission_fxn, init_mu, init_cov, trans_cov, obs_cov) 
            test_Z,  test_dataset   = createDataset(Ntest, Ttest, transition_fxn, emission_fxn, init_mu, init_cov, trans_cov, obs_cov) 
            savefile       = syntheticDIR+'/synthetic'+str(s)+'.h5' 
            h5file = h5py.File(savefile,mode='w')
            h5file.create_dataset('train_z', data=train_Z)
            h5file.create_dataset('test_z',  data=test_Z)
            h5file.create_dataset('valid_z', data=valid_Z)
            h5file.create_dataset('train',   data=train_dataset)
            h5file.create_dataset('test',    data=test_dataset)
            h5file.create_dataset('valid',   data=valid_dataset)
            h5file.close()
            print 'Created: ',savefile
    print 'REMEMBER TO RUN BASELINES!'

if __name__=='__main__':
    _processMNIST()
    _processBinarizedMNIST()
    _processPolyphonic('jsb')
    _processSynthetic('synthetic2')
