from __future__ import division
import six.moves.cPickle as pickle
from collections import OrderedDict
import numpy as np
import sys, time, os, gzip, theano,math
from theano import config
theano.config.compute_test_value = 'warn'
from theano.printing import pydotprint
import theano.tensor as T
from utils.misc import saveHDF5
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils.optimizer import adam,rmsprop
from __init__ import BaseModel
from datasets.synthp import params_synthetic
from datasets.synthpTheano import updateParamsSynthetic

class LogisticRegression(BaseModel, object):
    """
                                Logistic Regression (classify if a digit is less than or greater than 5) 
    """
    def __init__(self, params, paramFile=None, reloadFile=None):
        super(LogisticRegression,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile)
        assert self.params['nonlinearity']!='maxout','No support for maxout units in LR'
    def _fakeData(self):
        N = 2
        out   = np.random.random((N,1)).astype(config.floatX)
        small = out<0.5
        large = out>=0.5
        out[small] = 0.
        out[large]= 1.
        return np.random.random((N,self.params['dim_observations'])).astype(config.floatX), out
    
    def _createParams(self):
        npWeights  = OrderedDict()
        npWeights['W_lr'] = self._getWeight((self.params['dim_observations'], 1))
        npWeights['b_lr'] = self._getWeight((1,))
        return npWeights
    
    def _buildModel(self):
        self.updates_ack= True
        X             = T.matrix('X',   dtype=config.floatX)
        Y             = T.matrix('Y',   dtype=config.floatX)
        X.tag.test_value, Y.tag.test_value    = self._fakeData()
        #output_params_t= T.nnet.sigmoid(self._LinearNL(self.tWeights['W_lr'], self.tWeights['b_lr'], X, onlyLinear=True))
        output_params_t= T.nnet.sigmoid(self._BNlayer(self.tWeights['W_lr'], self.tWeights['b_lr'], X, validation=False, onlyLinear=True))
        nll_t          = T.nnet.binary_crossentropy(output_params_t, Y).sum()

        #output_params_e = T.nnet.sigmoid(self._LinearNL(self.tWeights['W_lr'], self.tWeights['b_lr'], X, onlyLinear=True))
        output_params_e= T.nnet.sigmoid(self._BNlayer(self.tWeights['W_lr'], self.tWeights['b_lr'], X, validation=True, onlyLinear=True))
        nll_e          = T.nnet.binary_crossentropy(output_params_e, Y).sum()
        
        if not self.params['validate_only']:
            model_params  = self._getModelParams()
            print len(self.updates),' extraneous updates'
            optimizer_up, norm_list  = self._setupOptimizer(nll_t, 
                                                        model_params, 
                                                        lr=self.params['lr'],
                                                        divide_grad = T.cast(X.shape[0],config.floatX))
            optimizer_up+=self.updates
            self.train      = theano.function([X,Y], [nll_t,self.tWeights['_lr_BN_running_mean'], self.tWeights['_lr_BN_running_var']], updates = optimizer_up)
        self.evaluate   = theano.function([X,Y],nll_e)
        
    def learn(self, dataset,labels, epoch_start=0, epoch_end=1000, batch_size=200, shuffle=False,
             savefreq=None, savefile = None, dataset_eval = None, labels_eval = None):
        """
                                    Loop through dataset for training
        """
        assert not self.params['validate_only'],'Cannot learn in validate mode'
        assert len(dataset.shape)==2,'Expecting 2D tensor'
        assert dataset.shape[1]==self.params['dim_observations'],'Dimension of observation not valid'
        
        N = dataset.shape[0]
        idxlist = range(N)
        if shuffle:
            np.random.shuffle(idxlist)
        nll_train_list,nll_valid_list = [],[]
        for epoch in range(epoch_start, epoch_end):
            start_time = time.time()
            nll = 0
            for bnum,st_idx in enumerate(range(0,N,batch_size)):
                end_idx = min(st_idx+batch_size, N)
                X       = dataset[idxlist[st_idx:end_idx],:].astype(config.floatX)
                Y       = labels[idxlist[st_idx:end_idx]][:,None].astype(config.floatX)
                batch_nll,running_mean,running_var = self.train(X=X, Y=Y)
                nll  += batch_nll
                self._p(('Bnum:%d, Batch Bound: %.4f, Running Mean :%.4f, Running Var: %.4f')%(bnum,batch_nll/float(X.shape[0]),
                    np.linalg.norm(running_mean), np.linalg.norm(running_var))) 
            nll /= float(dataset.shape[0])
            nll_train_list.append((epoch,nll))
            end_time   = time.time()
            if epoch%10==0:
                self._p(('(Ep %d) NLL: %.4f [Took %.4f seconds]')%(epoch, nll,end_time-start_time))
            if savefreq is not None and epoch%savefreq==0:
                self._p(('Saving at epoch %d'%epoch))
                if dataset_eval is not None and labels_eval is not None:
                    nll_valid_list.append((epoch,self.evaluateNLL(dataset_eval,labels_eval, batch_size=batch_size))) 
                intermediate = {}
                intermediate['train_nll'] = np.array(nll_train_list)
                intermediate['valid_nll'] = np.array(nll_valid_list)
        retMap = {}
        retMap['train_nll'] =  np.array(nll_train_list)
        retMap['valid_nll'] =  np.array(nll_valid_list)
        return retMap
    
    def evaluateNLL(self, dataset, labels, batch_size = 200):
        """
                                            Evaluate likelihood of dataset
        """
        nll = 0
        start_time = time.time()
        N   = dataset.shape[0]
        for bnum,st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size, N)
            X       = dataset[st_idx:end_idx,:].astype(config.floatX)
            Y       = labels[st_idx:end_idx][:,None].astype(config.floatX)
            batch_nll = self.evaluate(X=X, Y=Y)
            nll  += batch_nll
            self._p(('\tBnum:%d, Batch Bound: %.4f')%(bnum,batch_nll/float(X.shape[0]))) 
        nll /= float(X.shape[0])
        end_time   = time.time()
        self._p(('(Evaluation) NLL: %.4f [Took %.4f seconds]')%(nll,end_time-start_time))
        return nll

if __name__=='__main__':
    print 'Starting Logistic Regression'
    from datasets.load import loadDataset
    mnist = loadDataset('mnist')
    labels_train = (mnist['train_y']>=5.)*1.
    labels_test  = (mnist['test_y']>=5.)*1.
    params= {}
    from utils.parse_args_vae import params
    params['dim_observations'] = 784
    params['validate_only'] = False
    pfile = 'tmp'
    LR    = LogisticRegression(params,paramFile=pfile) 
    results = LR.learn(mnist['train'],labels_train, epoch_start=0, epoch_end=100, savefreq = 10, 
            batch_size=2000, dataset_eval=mnist['test'], labels_eval=labels_test)
    os.unlink(pfile)
    import ipdb;ipdb.set_trace()
