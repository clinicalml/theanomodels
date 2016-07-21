"""
Author: Rahul G. Krishnan
Divergences used between probability measures 
"""
import theano.tensor as T
def KLGaussian(mu_1,cov_1,mu_2,cov_2,logCov=False):
    """
    Estimate the KL divergence between two gaussians with diagonal covariance
    KL(q||p)=0.5*(log|Sigma_2|-log|Sigma_1| -D + tr(Sigma_2^-1 Sigma_1) + (mu_2-mu_1)^T Sigma_2^-1 (mu_2-mu_1)) 
    """
    if type(mu_1) is list:
        assert type(mu_2) is list and type(cov_1) is list and type(cov_2) is list,'Expecting lists'
        div = 0 
        for m1,c1,m2,c2 in zip(mu_1,cov_1,mu_2,cov_2):
            div+=KLGaussian(m1,c1,m2,c2,logCov=logCov)
        return div
    else:
        diff   = mu_2-mu_1
        if logCov:
            logcov_1 = cov_1
            logcov_2 = cov_2
            return 0.5*(logcov_2-logcov_1-1+T.exp(logcov_1-logcov_2)+(diff**2)*T.exp(-1*logcov_2)).sum()
        else:
            return 0.5*(T.log(cov_2)-T.log(cov_1)-1+cov_1/cov_2+(diff**2)/cov_2).sum()

def BhattacharryaGaussian(mu_1,cov_1,mu_2,cov_2,logCov=False):
    """
    Estimate the Bhattacharyya distance between two gaussians with diagonal covariance
    D_B(q||p)
    See: http://like.silk.to/studymemo/PropertiesOfMultivariateGaussianFunction.pdf
    https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    if type(mu_1) is list:
        assert type(mu_2) is list and type(cov_1) is list and type(cov_2) is list,'Expecting lists'
        div = 0 
        for m1,c1,m2,c2 in zip(mu_1,cov_1,mu_2,cov_2):
            div+=BhattacharryaGaussian(m1,c1,m2,c2,logCov=logCov)
        return div
    else:
        diff = mu_1-mu_2
        if logCov:
            logcov_1= cov_1
            logcov_2= cov_2
            P       = (T.exp(logcov_2)+T.exp(logcov_1))/2
            logdet  = T.log(P) - 0.5*(logcov_2+logcov_1)
        else:
            P       = (cov_2+cov_1)/2
            logdet  =T.log(P) -0.5*T.log(cov_2*cov_1)
        D_B  = (1/8.)*(diff**2)/P + (1/2.)*logdet 
        return D_B.sum()
