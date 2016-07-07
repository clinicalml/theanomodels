"""
Author: Rahul G. Krishnan
File containing divergences used between probability measures 
"""
import theano.tensor as T
def KLGaussian(mu_1,cov_1,mu_2,cov_2):
    """
    Estimate the KL divergence between two gaussians with diagonal covariance
    KL(q||p) 0.5*(log|Sigma_2| - log |Sigma_1| 
    """
    diff   = mu_2-mu_1
    return 0.5*(T.log(cov_2)-T.log(cov_1)-1+cov_1/cov_2+(diff**2)/cov_2).sum()

def BhattacharryaGaussian(mu_1,cov_1,mu_2,cov_2):
    """
    Estimate the Bhattacharyya distance between two gaussians with diagonal covariance
    D_B(q||p)
    See: http://like.silk.to/studymemo/PropertiesOfMultivariateGaussianFunction.pdf
    """
    diff = mu_1-mu_2
    P    = (cov_2+cov_1)/2
    D_B  = (1/8.)*(diff**2)/P + (1/2.)*(T.log(P) -0.5*T.log(cov_2)-0.5*T.log(cov_1))
    return D_B.sum()
