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

class LSTM(BaseModel, object):
    """
                                LONG SHORT TERM MEMORY MODEL (f)
    """
    def __init__(self, params, paramFile=None, reloadFile=None):
        super(LSTM,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile)
        assert self.params['nonlinearity']!='maxout','No support for maxout units in LSTM'
    def _fakeData(self):
        T = 3
        N = 2
        mask = np.random.random((N,T)).astype(config.floatX)
        small = mask<0.5
        large = mask>=0.5
        mask[small] = 0.
        mask[large]= 1.
        return np.random.random((N,T,self.params['dim_observations'])).astype(config.floatX),mask
    
    def _createParams(self):
        """
                                        Create parameters
        """
        npWeights = OrderedDict()
        DIM_HIDDEN = self.params['rnn_size']
        npWeights['W_input'] = self._getWeight((self.params['dim_observations'], DIM_HIDDEN))
        npWeights['b_input'] = self._getWeight((DIM_HIDDEN,))
        for l in range(self.params['rnn_layers']):
            npWeights['W_lstm_l_'+str(l)] = self._getWeight((DIM_HIDDEN,
                                                           DIM_HIDDEN*4),scheme='lstm')
            npWeights['b_lstm_l_'+str(l)] = self._getWeight((DIM_HIDDEN*4,),scheme='lstm')
            npWeights['U_lstm_l_'+str(l)] = self._getWeight((DIM_HIDDEN,
                                                           DIM_HIDDEN*4),scheme='lstm')
        npWeights['W_output'] = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
        npWeights['b_output'] = self._getWeight((self.params['dim_observations'],))
        return npWeights
    
    def _setupLSTM(self, X, dropout_prob = 0.):
        """
                                        Setup LSTM RNN 
        """
        #X_embed      = self._BNlayer(self.tWeights['W_input'],self.tWeights['b_input'],X,validation=dropout_prob==0.)
        X_embed      = self._LinearNL(self.tWeights['W_input'],self.tWeights['b_input'],X)
        lstm_output  = self._LSTMlayer(X_embed, 'l', dropout_prob = dropout_prob)
        lstm_output  = lstm_output.swapaxes(0,1)
        params_embed = T.dot(lstm_output, self.tWeights['W_output'])+self.tWeights['b_output']
        output_params= T.nnet.sigmoid(params_embed)
        return output_params
    
    def _buildModel(self):
        """
                                        Build Model
        """
        #Expecting (X: batch_size x T x input_dim)
        X             = T.tensor3('X',   dtype=config.floatX)
        M             = T.matrix('M',    dtype=config.floatX)
        X.tag.test_value, M.tag.test_value    = self._fakeData()
        #Pad with start token
        input_X       = T.concatenate([T.alloc(np.asarray(0., dtype=config.floatX),X.shape[0],1,X.shape[2]), 
                                 X[:,:-1,:]],axis=1)
        output_params_t = self._setupLSTM(input_X, dropout_prob = self.params['rnn_dropout'])
        output_params_e = self._setupLSTM(input_X, dropout_prob = 0.)
        
        nll_t         = (T.nnet.binary_crossentropy(output_params_t, X).sum(2)*M).sum()
        nll_e         = (T.nnet.binary_crossentropy(output_params_e, X).sum(2)*M).sum()
        
        #Get updates from optimizer
        #Create theano functions for different purposes
        if not self.params['validate_only']:
            model_params  = self._getModelParams()
            optimizer_up, norm_list  = self._setupOptimizer(nll_t, 
                                                        model_params, 
                                                        lr=self.params['lr'],
                                                        reg_type =self.params['reg_type'], 
                                                        reg_spec =self.params['reg_spec'], 
                                                        reg_value= self.params['reg_value'],
                                                        grad_norm = 1.,
                                                        divide_grad = M.sum())
            self.updates_ack= True
            self._p('Added '+str(len(self.updates))+' updates')
            optimizer_up+=self.updates
             
            self.train      = theano.function([X, M], nll_t, updates = optimizer_up)
            self.train_debug= theano.function([X, M],[nll_t,norm_list[0],norm_list[1],norm_list[2]],updates = optimizer_up)
        self.evaluate   = theano.function([X, M],nll_e)
        
        
    def learn(self, dataset, mask, epoch_start=0, epoch_end=1000, batch_size=200, shuffle=False,
             savefreq=None, savefile = None, dataset_eval = None, mask_eval = None):
        """
                                    Loop through dataset for training
        """
        assert not self.params['validate_only'],'Cannot learn in validate mode'
        assert len(dataset.shape)==3,'Expecting 3D tensor'
        assert dataset.shape[2]==self.params['dim_observations'],'Dimension of observation not valid'
        
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
                X       = dataset[idxlist[st_idx:end_idx],:,:].astype(config.floatX)
                M       = mask[idxlist[st_idx:end_idx],:].astype(config.floatX)
                batch_nll = self.train(X=X, M=M)
                nll  += batch_nll
                self._p(('Bnum:%d, Batch Bound: %.4f')%(bnum,batch_nll/float(M.sum()))) 
            nll /= float(mask.sum())
            nll_train_list.append((epoch,nll))
            end_time   = time.time()
            if bnum%10==0:
                self._p(('(Ep %d) NLL: %.4f [Took %.4f seconds]')%(epoch, nll,end_time-start_time))
            if savefreq is not None and epoch%savefreq==0:
                assert savefile is not None, 'expecting savefile'
                self._p(('Saving at epoch %d'%epoch))
                self._saveModel(fname = savefile+'-EP'+str(epoch))
                if dataset_eval is not None and mask_eval is not None:
                    nll_valid_list.append((epoch,self.evaluateNLL(dataset_eval, mask_eval, batch_size=batch_size))) 
                intermediate = {}
                intermediate['train_nll'] = np.array(nll_train_list)
                intermediate['valid_nll'] = np.array(nll_valid_list)
                saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
        retMap = {}
        retMap['train_nll'] =  np.array(nll_train_list)
        retMap['valid_nll'] =  np.array(nll_valid_list)
        return retMap
    
    def evaluateNLL(self, dataset, mask, batch_size = 200):
        """
                                            Evaluate likelihood of dataset
        """
        nll = 0
        start_time = time.time()
        N   = dataset.shape[0]
        for bnum,st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size, N)
            X       = dataset[st_idx:end_idx,:,:].astype(config.floatX)
            M       = mask[st_idx:end_idx,:].astype(config.floatX)
            batch_nll = self.evaluate(X=X, M=M)
            nll  += batch_nll
            self._p(('\tBnum:%d, Batch Bound: %.4f')%(bnum,batch_nll/float(M.sum()))) 
        nll /= float(mask.sum())
        end_time   = time.time()
        self._p(('(Evaluation) NLL: %.4f [Took %.4f seconds]')%(nll,end_time-start_time))
        return nll
