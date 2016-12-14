import six.moves.cPickle as pickle
from collections import OrderedDict
import sys, time, os
import numpy as np
import gzip
import theano
from theano import config
theano.config.compute_test_value = 'warn'
from theano.printing import pydotprint
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from ..utils.optimizer import adam,rmsprop
from ..utils.misc import saveHDF5
from . import BaseModel

class MLP(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None):
        super(MLP,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile)
    def _createParams(self):
        """
                    _createParams: create parameters necessary for the model
        """
        npWeights = OrderedDict()
        if 'dim_hidden' not in self.params:
            self.params['dim_hidden']= dim_hidden
        DIM_HIDDEN = self.params['dim_hidden']
        #Weights in recognition network model
        for layer in range(self.params['nlayers']):
            dim_input     = DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if layer==0:
                dim_input     = self.params['dim_observations']
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            npWeights['layer_'+str(layer)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['layer_'+str(layer)+'_b'] = self._getWeight((dim_output, ),scheme='zero')
        npWeights['layer_out_W'] = self._getWeight((dim_input,self.params['nclasses']))
        npWeights['layer_out_b'] = self._getWeight((self.params['nclasses'], ),scheme='zero')
        return npWeights
    
    def _fakeData(self):
        """
                                Compile all the fake data 
        """
        X = np.random.rand(2, self.params['dim_observations']).astype(config.floatX)
        m1=X>0.5
        m0=X<=0.5
        X[m0]=0
        X[m1]=1
        Y = np.random.randint(low=0,high=2,size=2).astype('int32')#.astype(config.floatX)
        #Y = np.random.multinomial(n=1,pvals=[0.5,0.5],size=2).astype(config.floatX)
        return X, Y

    def _buildClassifier(self, X, dropout_prob=0., normlayer=None, evaluation=False):
        """
                                Build classifier 
        """
        #Dropout
        self._p(('Classifier with dropout :%.4f')%(dropout_prob))
        inp = self._dropout(X,dropout_prob)
        #Hidden layers
        for layer in range(self.params['nlayers']):
            W = self.tWeights['layer_'+str(layer)+'_W']
            b = self.tWeights['layer_'+str(layer)+'_b']
            if normlayer is None:
                inp = self._LinearNL(W,b,inp)
            elif normlayer=='batchnorm':
                inp = self._BNlayer(W,b,inp,evaluation=evaluation)
            elif normlayer=='layernorm':
                inp = self._LayerNorm(W,b,inp)
            else:
                try:
                    errmsg = "unknown normalization layer type: %s" % normlayer
                except:
                    errmsg = "unknown normalization layer type"
                assert False, errmsg

        #Output layer
        inp = self._LinearNL(self.tWeights['layer_out_W'],self.tWeights['layer_out_b'],inp,onlyLinear=True)
        probs = T.nnet.softmax(inp)

        return probs
    
    def _buildModel(self):
        """
                        ******BUILD MLP GRAPH******
        """
        #Inputs to graph
        X   = T.matrix('X',   dtype=config.floatX)
        Y   = T.ivector('Y')#,   dtype=config.floatX)
        X.tag.test_value, Y.tag.test_value  = self._fakeData()
        self.updates_ack = True
        #Learning Rates and annealing objective function
        #Add them to npWeights/tWeights to be tracked [do not have a prefix _W or _b so wont be diff.]
        self._addWeights('lr', np.asarray(self.params['lr'],dtype=config.floatX),borrow=False)
        self._addWeights('update_ctr', np.asarray(1.,dtype=config.floatX),borrow=False)
        
        lr  = self.tWeights['lr']
        iteration_t    = self.tWeights['update_ctr'] 
        
        lr_update = [(lr,T.switch(lr/1.0005<1e-4,lr,lr/1.0005))]
        
        #Build training graph
        probs_t = self._buildClassifier(X,self.params['input_dropout'],self.params['normlayer'],evaluation=False)
        probs_e = self._buildClassifier(X,0.,self.params['normlayer'],evaluation=True)
        
        #Cost function to minimize
        crossentropy_train = T.nnet.categorical_crossentropy(probs_t,Y).sum()
        crossentropy_eval = T.nnet.categorical_crossentropy(probs_e,Y).sum()

        #Accuracy
        ncorrect_train = T.eq(T.argmax(probs_t,axis=1),Y).sum()
        ncorrect_eval = T.eq(T.argmax(probs_e,axis=1),Y).sum()

        #Optimizer with specification for regularizer
        model_params         = self._getModelParams()
        optimizer_up, norm_list  = self._setupOptimizer(crossentropy_train, model_params,lr = lr, 
                                                        reg_type =self.params['reg_type'], 
                                                        reg_spec =self.params['reg_spec'], 
                                                        reg_value= self.params['reg_value'],
                                                       grad_norm = self.params['grad_norm'],
                                                       divide_grad = T.cast(X.shape[0],config.floatX))

        #self.updates is container for all updates (e.g. see _BNlayer in __init__.py)
        self.updates += optimizer_up
        
        #Build theano functions
        fxn_inputs = [X,Y]
        
        #Importance sampled estimate
        self.train      = theano.function(fxn_inputs, [crossentropy_train, ncorrect_train],  
                                              updates = self.updates, name = 'Train')
        self.debug      = theano.function(fxn_inputs, [crossentropy_train,norm_list[0],
                                                       norm_list[1],norm_list[2],ncorrect_train],  
                                              updates = self.updates, name = 'Train+Debug')
        self.evaluate   = theano.function(fxn_inputs, [crossentropy_eval, ncorrect_eval], name = 'Evaluate')
        self.decay_lr   = theano.function([],lr.sum(),name = 'Update LR',updates=lr_update)

    def sampleDataset(self, dataset):
        p = np.random.uniform(low=0,high=1,size=dataset.shape)
        return (dataset >= p).astype(config.floatX)

    def evaluateClassifier(self, dataset, batch_size):
        """
                         Evaluate neg log likelihood and accuracy
        """
        N = dataset['X'].shape[0]
        crossentropy = 0
        ncorrect = 0
        for bnum, st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size,N)
            X = self.sampleDataset(dataset['X'][st_idx:end_idx].astype(config.floatX))
            Y = dataset['Y'][st_idx:end_idx].astype('int32')
            batch_crossentropy, batch_ncorrect = self.evaluate(X=X,Y=Y)
            crossentropy += batch_crossentropy
            ncorrect += batch_ncorrect
        crossentropy /= float(N)
        accuracy = ncorrect/float(N)
        return crossentropy, accuracy

    def learn(self, dataset, epoch_start=0, epoch_end=1000, batch_size=200, shuffle=True, 
              savefile = None, savefreq = None, evalfreq = 1, dataset_eval=None):
        assert len(dataset['X'].shape)==2,'Expecting 2D dataset matrix'
        assert dataset['X'].shape[1]==self.params['dim_observations'],'dim observations incorrect'
        assert dataset['X'].shape[0]==dataset['Y'].shape[0],'len X and Y do not match'
        assert shuffle==True,'Shuffle should be true, especially when using batchnorm'
        N = dataset['X'].shape[0]
        idxlist = range(N)
        train_crossentropy,train_accuracy,valid_crossentropy,valid_accuracy,current_lr = [],[],[],[],self.params['lr']
        #Training epochs
        for epoch in range(epoch_start, epoch_end+1):
            start_time = time.time()
            crossentropy, ncorrect, grad_norm, param_norm = 0, 0, [], []
            
            #Update learning rate
            #current_lr = self.decay_lr()

            if shuffle:
                np.random.shuffle(idxlist)

            #Go through dataset
            for bnum,st_idx in enumerate(range(0,N,batch_size)):
                end_idx = min(st_idx+batch_size, N)
                X       = self.sampleDataset(dataset['X'][idxlist[st_idx:end_idx]].astype(config.floatX))
                Y       = dataset['Y'][idxlist[st_idx:end_idx]].astype('int32')
                Nbatch  = X.shape[0]
                
                #Forward/Backward pass
                batch_crossentropy, batch_ncorrect = self.train(X=X, Y=Y)
                
                crossentropy  += batch_crossentropy
                ncorrect += batch_ncorrect
                
            crossentropy /= float(N)
            accuracy = ncorrect/float(N)
            train_crossentropy.append((epoch,crossentropy))
            train_accuracy.append((epoch,accuracy))
            end_time   = time.time()
            self._p(('Ep (%d) Train CE_NLL: %.4f, Train Accuracy: %0.4f [Took %.4f seconds]')%(epoch, crossentropy, accuracy, (end_time-start_time)))
            
            if evalfreq is not None and epoch%evalfreq==0:
                if dataset_eval is not None:
                    v_crossentropy, v_accuracy = self.evaluateClassifier(dataset_eval, batch_size=batch_size)
                    valid_crossentropy.append((epoch,v_crossentropy))
                    valid_accuracy.append((epoch,v_accuracy))
                    self._p(('Ep (%d): Valid CE_NLL: %.4f, Valid Accuracy: %.4f')%(epoch, v_crossentropy, v_accuracy))

            if savefreq is not None and epoch%savefreq==0:
                self._p(('Saving at epoch %d'%epoch))
                self._saveModel(fname=savefile+'EP'+str(epoch))
                intermediate = {}
                intermediate['train_crossentropy'] = np.array(train_crossentropy)
                intermediate['train_accuracy'] = np.array(train_accuracy)
                intermediate['valid_crossentropy'] = np.array(valid_crossentropy)
                intermediate['valid_accuracy'] = np.array(valid_accuracy)
                saveHDF5(savefile+'EP'+str(epoch)+'-stats.h5', intermediate)
        ret_map = {}
        ret_map['train_crossentropy'] = np.array(train_crossentropy)
        ret_map['train_accuracy'] = np.array(train_accuracy)
        ret_map['valid_crossentropy'] = np.array(valid_crossentropy)
        ret_map['valid_accuracy'] = np.array(valid_accuracy)
        return ret_map
