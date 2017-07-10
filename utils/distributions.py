"""
Classes for sampling distributions
"""
class GMM(object):
    """ Gaussian Mixture Model """
    def __init__(self, mu, logcov, pi_unnorm):
        """
        mu:     K x dim_z - K means
        logcov: K x dim_z - K logcovariance matrices (assumed diagonal)
        pi:     Mixture proportions
        """
        self.mu     = mu
        self.logcov = logcov
        self.pi     = pi_unnorm/float(pi_unnorm.sum())
        self.K      = self.pi.shape[0]

    def ll(self, z, assignment = None):
        """Log-likelihood of z under model
        z: N x ds
        return: N x 1
        """
        #likelihood under each of the mixture components
        # N x K x dz
        N   = z.shape[0]
        z_k = z.reshape(N,1,-1).repeat(self.K,1)
        #N x K = N x K x dz
        ll_k = ll_gaussian(z_k, self.mu[None,:,:].repeat(N,0), self.logcov[None,:,:].repeat(N,0))
        if assignment is None:
            logpi= np.log(self.pi.reshape(1,-1).repeat(N,0))
            x_n  = logpi+ll_k
            #N x 1
            #compute likelihood under each of the Gaussians 
            ll   = logsumexp(x_n)
        else:
            assert assignment.shape[0]==z.shape[0],'Check shapes'
            ll   = ll_k[np.arange(N), assignment]
        return ll
    
    def sample(self, N, debug = None):
        """Sample from model
        N: (int)
        return: sample N x ds
        """
        #return a point sampled from the model
        K,dz   = self.K, self.mu.shape[1]
        #Sample N times from each of the K mixture components
        eps  = np.random.randn(N,K,dz)
        #replicate the mean/logcov N times - sample from each of the mixtures 
        z    = self.mu[None,:,:].repeat(N,0) + eps*np.exp(0.5*self.logcov[None,:,:].repeat(N,0))
        #Pick the sample
        probs= np.argmax(np.random.multinomial(1, self.pi, size=(N,)),axis=1)
        if type(debug) is dict:
            debug['probs'] = probs
            debug['z']     = z
        z_s  = z[np.arange(N), probs, :]
        return z_s

class Gaussian(object):
    """ Gaussian Distribution """
    def __init__(self, mu, logcov):
        """
        mu:     (ds,) - K means
        logcov: (ds,) - K logcovariance matrices (assumed diagonal)
        """
        self.mu     = mu.squeeze()
        self.logcov = logcov.squeeze()

    def ll(self, z):
        """Log-likelihood of z under model
        z: N x ds
        return: N x 1
        """
        #likelihood under the model
        N   = z.shape[0]
        z_k = z.reshape(N,1,-1)
        ll = ll_gaussian(z_k, self.mu.reshape(1,1,-1).repeat(N,0), self.logcov.reshape(1,1,-1).repeat(N,0))
        return ll
    
    def sample(self, N):
        """Sample from model
        N: (int)
        return: sample N x ds
        """
        #return N points sampled from the model
        z    = self.mu[None,:].repeat(N,0) + eps*np.exp(0.5*self.logcov[None,:].repeat(N,0))
        return z
