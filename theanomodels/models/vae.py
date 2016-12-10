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
from .__init__ import BaseModel

class VAE(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None):
        super(VAE,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile)
    def _createParams(self):
        """
                    _createParams: create parameters necessary for the model
        """
        npWeights = OrderedDict()
        if 'q_dim_hidden' not in self.params or 'p_dim_hidden' not in self.params:
            self.params['q_dim_hidden']= dim_hidden
            self.params['p_dim_hidden']= dim_hidden
        DIM_HIDDEN = self.params['q_dim_hidden']
        #Weights in recognition network model
        for q_l in range(self.params['q_layers']):
            dim_input     = DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if q_l==0:
                dim_input     = self.params['dim_observations']
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            npWeights['q_'+str(q_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['q_'+str(q_l)+'_b'] = self._getWeight((dim_output, ))
        if self.params['inference_model']=='single':
            npWeights['q_mu_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
            npWeights['q_logcov_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
            npWeights['q_mu_b']     = self._getWeight((self.params['dim_stochastic'],))
            npWeights['q_logcov_b'] = self._getWeight((self.params['dim_stochastic'],))
        else:
            assert False,'Invalid variational model'
        #Generative Model
        DIM_HIDDEN = self.params['p_dim_hidden']
        for p_l in range(self.params['p_layers']):
            dim_input     = DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if p_l==0:
                dim_input     = self.params['dim_stochastic']
            if self.params['nonlinearity']=='maxout':
                dim_output= DIM_HIDDEN*self.params['maxout_stride']
            npWeights['p_'+str(p_l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['p_'+str(p_l)+'_b'] = self._getWeight((dim_output, ))
        if self.params['data_type']=='real':
            npWeights['p_mu_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_logcov_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_mu_b']     = self._getWeight((self.params['dim_observations'],))
            npWeights['p_logcov_b'] = self._getWeight((self.params['dim_observations'],))
        else:
            npWeights['p_mean_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_mean_b']     = self._getWeight((self.params['dim_observations'],))
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
        eps = np.random.randn(2, self.params['dim_stochastic']).astype(config.floatX)
        return X,eps
    
    def _buildEmission(self, z, X, add_noise = False):
        """
                                Build subgraph to estimate conditional params
        """
        if add_noise:
            inp_p   = z + self.srng.normal(z.shape,0,0.0025,dtype=config.floatX)
        else:
            inp_p   = z
        for p_l in range(self.params['p_layers']):
            inp_p = self._LinearNL(self.tWeights['p_'+str(p_l)+'_W'], self.tWeights['p_'+str(p_l)+'_b'], inp_p)

        if self.params['data_type']=='real':
            mu_p    = self._LinearNL(self.tWeights['p_mu_W'],self.tWeights['p_mu_b'],inp_p, onlyLinear=True)
            logcov_p= self._LinearNL(self.tWeights['p_logcov_W'],self.tWeights['p_logcov_b'],inp_p, onlyLinear=True)
            negCLL_m= 0.5 * (np.log(2 * np.pi) + logcov_p + ((X - mu_p) / T.exp(0.5*logcov_p))**2)
            return (mu_p, logcov_p), negCLL_m.sum(1,keepdims=True)
        else:
            mean_p = T.nnet.sigmoid(self._LinearNL(self.tWeights['p_mean_W'],self.tWeights['p_mean_b'],inp_p,onlyLinear=True))
            negCLL_m = T.nnet.binary_crossentropy(mean_p,X)
            return (mean_p,), negCLL_m.sum(1,keepdims=True)
    
    def _buildInference(self, X, dropout_prob = 0.):
        """
                                Build subgraph to do inference 
        """
        self._p(('Inference with dropout :%.4f')%(dropout_prob))
        inp = self._dropout(X,dropout_prob)
        for q_l in range(self.params['q_layers']):
            inp = self._LinearNL(self.tWeights['q_'+str(q_l)+'_W'], self.tWeights['q_'+str(q_l)+'_b'], inp)
       
        mu      = self._LinearNL(self.tWeights['q_mu_W'],self.tWeights['q_mu_b'],inp,onlyLinear=True)
        logcov  = self._LinearNL(self.tWeights['q_logcov_W'],self.tWeights['q_logcov_b'],inp,onlyLinear=True)
        return mu, logcov
    
    def _evaluateKL(self, mu, logcov, eps):
        """
                            KL divergence between N(0,I) and N(mu,exp(logcov))
        """
        #Pass z back
        z, KL = None, None
        if self.params['inference_model']=='single':
            z       = mu + T.exp(0.5*logcov)*eps
            KL      = 0.5*T.sum(-logcov -1 + T.exp(logcov) +mu**2 )
        else:
            assert False,'Invalid inference model: '+self.params['inference_model']
        return z,KL
    
    
    def _buildModel(self):
        """
                                       ******BUILD VAE GRAPH******
        """
        #Inputs to graph
        X   = T.matrix('X',   dtype=config.floatX)
        eps = T.matrix('eps', dtype=config.floatX)
        X.tag.test_value, eps.tag.test_value   = self._fakeData()
        self.updates_ack = True
        #Learning Rates and annealing objective function
        #Add them to npWeights/tWeights to be tracked [do not have a prefix _W or _b so wont be diff.]
        self._addWeights('lr', np.asarray(self.params['lr'],dtype=config.floatX),borrow=False)
        self._addWeights('anneal', np.asarray(0.01,dtype=config.floatX),borrow=False)
        self._addWeights('update_ctr', np.asarray(1.,dtype=config.floatX),borrow=False)
        
        lr  = self.tWeights['lr']
        anneal         = self.tWeights['anneal']
        iteration_t    = self.tWeights['update_ctr'] 
        
        lr_update = [(lr,T.switch(lr/1.0005<1e-4,lr,lr/1.0005))]
        anneal_div     = 10000.
        anneal_update  = [(iteration_t, iteration_t+1),
                          (anneal,T.switch(0.01+iteration_t/anneal_div>1,1.,0.01+iteration_t/anneal_div))]
        
        #Build training graph
        mu_t, logcov_t     = self._buildInference(X,self.params['input_dropout'])
        z_t,  KL_t         = self._evaluateKL(mu_t, logcov_t, eps)
        _, negCLL_t = self._buildEmission(z_t, X, add_noise = True)
        
        meanAbsDev = 0
        #Build evaluation graph
        mu_e, logcov_e     = self._buildInference(X,0.)
        z_e,  KL_e         = self._evaluateKL(mu_e, logcov_e, eps)
        params_e, negCLL_e = self._buildEmission(z_e, X, add_noise = False)
        
        #Cost function to minimize
        upperbound_train     = negCLL_t.sum()+anneal*KL_t
        upperbound_eval      = negCLL_e.sum()+KL_e
        
        
        #Optimizer with specification for regularizer
        model_params         = self._getModelParams()
        optimizer_up, norm_list  = self._setupOptimizer(upperbound_train-meanAbsDev, model_params,lr = lr, 
                                                        reg_type =self.params['reg_type'], 
                                                        reg_spec =self.params['reg_spec'], 
                                                        reg_value= self.params['reg_value'],
                                                       grad_norm = 1.,
                                                       divide_grad = T.cast(X.shape[0],config.floatX))
        #Also add annealing updates 
        optimizer_up+=anneal_update
        
        #Build theano functions
        fxn_inputs = [X,eps]
        
        #Importance sampled estimate
        ll_prior             = self._llGaussian(z_e, T.zeros_like(z_e,dtype=config.floatX),
                                                    T.zeros_like(z_e,dtype=config.floatX))
        ll_posterior         = self._llGaussian(z_e, mu_e, logcov_e)
        ll_estimate          = -1*negCLL_e+ll_prior.sum(1,keepdims=True)-ll_posterior.sum(1,keepdims=True)
        self.likelihood      = theano.function(fxn_inputs,ll_estimate)
        
        self.train      = theano.function(fxn_inputs, [upperbound_train,anneal.sum()],  
                                              updates = optimizer_up, name = 'Train')
        self.debug      = theano.function(fxn_inputs, [upperbound_train,norm_list[0],
                                                       norm_list[1],norm_list[2],anneal.sum()],  
                                              updates = optimizer_up, name = 'Train+Debug')
        self.inference  = theano.function(fxn_inputs, [z_e, mu_e, logcov_e], name = 'Inference')
        self.evaluate   = theano.function(fxn_inputs, upperbound_eval, name = 'Evaluate')
        self.decay_lr   = theano.function([],lr.sum(),name = 'Update LR',updates=lr_update)
        self.reconstruct= theano.function([z_e], list(params_e), name='Reconstruct')
        self.reset_anneal=theano.function([],anneal.sum(), updates = [(anneal,0.01)], name='reset anneal')
    def sample(self,nsamples=100):
        """
                                Sample from Generative Model
        """
        z = np.random.randn(nsamples,self.params['dim_stochastic']).astype(config.floatX)
        return self.reconstruct(z)
    
    def infer(self, data):
        """
                                Posterior Inference using recognition network
        """
        assert len(data.shape)==2,'Expecting 2D data matrix'
        assert data.shape[1]==self.params['dim_observations'],'Wrong dimensions for observations'
        
        eps  = np.random.randn(data.shape[0],self.params['dim_stochastic']).astype(config.floatX)
        return self.inference(X=data.astype(config.floatX),eps=eps)

    def evaluateBound(self, dataset, batch_size, S=10):
        """
                                    Evaluate bound S times on dataset 
        """
        N = dataset.shape[0]
        bound = 0
        for bnum,st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size, N)
            X       = dataset[st_idx:end_idx].astype(config.floatX)
            for s in range(S):
                eps     = np.random.randn(X.shape[0],self.params['dim_stochastic']).astype(config.floatX)
                if self.params['inference_model']=='single':
                    batch_bound = self.evaluate(X=X, eps=eps)
                else:
                    assert False,'Should not be here'
                bound  += batch_bound
        bound /= float(N*S)
        return bound
    
    def meanSumExp(self,mat,axis=1):
        """
        Estimate log 1/S \sum_s exp[ log k ] in a numerically stable manner where axis represents the sum
        """
        a = np.max(mat, axis=1, keepdims=True)
        return a + np.log(np.mean(np.exp(mat-a.repeat(mat.shape[1],1)),axis=1,keepdims=True))
    
    def impSamplingNLL(self, dataset, batch_size, S = 200):
        """
                                    Importance sampling based log likelihood
        """
        N = dataset.shape[0]
        ll = 0
        for bnum,st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size, N)
            X       = dataset[st_idx:end_idx].astype(config.floatX)
            
            batch_lllist = []
            for s in range(S):
                eps     = np.random.randn(X.shape[0],self.params['dim_stochastic']).astype(config.floatX)
                if self.params['inference_model']=='single':
                    batch_ll = self.likelihood(X=X, eps=eps)
                else:
                    assert False,'Should not be here'
                batch_lllist.append(batch_ll)
            ll  += self.meanSumExp(np.concatenate(batch_lllist,axis=1), axis=1).sum()
        ll /= float(N)
        return -ll
    
    def learn(self, dataset, epoch_start=0, epoch_end=1000, batch_size=200, shuffle=False, 
              savefile = None, savefreq = None, dataset_eval=None, replicate_K = None):
        assert len(dataset.shape)==2,'Expecting 2D dataset matrix'
        assert dataset.shape[1]==self.params['dim_observations'],'dim observations incorrect'
        N = dataset.shape[0]
        idxlist = range(N)
        if shuffle:
            np.random.shuffle(idxlist)
        trainbound,validbound,validll,current_lr = [],[],[],self.params['lr']
        #Training epochs
        for epoch in range(epoch_start, epoch_end+1):
            start_time = time.time()
            bound, grad_norm, param_norm = 0, [], []
            
            #Update learning rate
            current_lr = self.decay_lr()

            #Go through dataset
            for bnum,st_idx in enumerate(range(0,N,batch_size)):
                end_idx = min(st_idx+batch_size, N)
                X       = dataset[idxlist[st_idx:end_idx]].astype(config.floatX)
                Nbatch  = X.shape[0]
                if replicate_K is not None:
                    X   = X.repeat(replicate_K,0)
                eps     = np.random.randn(X.shape[0],self.params['dim_stochastic']).astype(config.floatX)
                
                #Forward/Backward pass
                if self.params['inference_model']=='single':
                    batch_bound, anneal = self.train(X=X, eps=eps)
                else:
                    assert False,'Should not be here'
                
                #Divide value of the bound by replicateK
                if replicate_K is not None:
                    batch_bound /= float(replicate_K)
                
                bound  += batch_bound
                if bnum%50==0:
                    self._p(('--Batch: %d, Batch Bound: %.4f, Anneal : %.4f, Lr : %.6e--')%
                            (bnum, batch_bound/Nbatch, anneal, current_lr))
                
            bound /= float(N)
            trainbound.append((epoch,bound))
            end_time   = time.time()
            self._p(('Ep (%d) Upper Bound: %.4f [Took %.4f seconds]')%(epoch, bound, (end_time-start_time)))
            
            if savefreq is not None and epoch%savefreq==0:
                self._p(('Saving at epoch %d'%epoch))
                self._saveModel(fname=savefile+'-EP'+str(epoch))
                if dataset_eval is not None:
                    v_bound = self.evaluateBound(dataset_eval, batch_size=batch_size)
                    validbound.append((epoch,v_bound))
                    v_ll = self.impSamplingNLL(dataset_eval, batch_size=batch_size)
                    validll.append((epoch,v_ll))
                    self._p(('Ep (%d): Valid Bound: %.4f, Valid LL: %.4f')%(epoch, v_bound, v_ll))
                intermediate = {}
                intermediate['train_bound'] = np.array(trainbound)
                intermediate['valid_bound'] = np.array(validbound)
                intermediate['valid_ll']    = np.array(validll)
                intermediate['samples']     = self.sample()
                saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
        ret_map={}
        ret_map['train_bound'] = np.array(trainbound)
        ret_map['valid_bound'] = np.array(validbound)
        ret_map['valid_ll']    = np.array(validll)
        return ret_map
