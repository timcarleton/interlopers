import numpy as np
from scipy.special import comb,gamma,gammainc,perm
from scipy.optimize import fsolve
from scipy.stats import expon,poisson

table=np.loadtxt('percentiletable.txt')

maxk=100
maxdelta=10

def normalize(lambdac,nobs):
    ks=np.array(range(nobs+1)).astype(float)

    if nobs>maxk:
        firstks=np.array(range(maxk)).astype(float)
        return np.sum(perm(nobs,firstks)/lambdac**firstks)
    else:
        return np.sum(perm(nobs,ks)/lambdac**ks)

        

def pdf(lambdatrue,lambdac,nobs):
    norm=normalize(lambdac,nobs)
    return (lambdatrue/lambdac+1)**nobs*np.exp(-lambdatrue)/norm

def cdf(lambdatrue,lambdac,nobs):
    ks=np.array(range(nobs+1)).astype(float)
    norm=normalize(lambdac,nobs)
    if nobs>maxk:
        firstks=np.array(range(maxk)).astype(float)
        return np.sum(perm(nobs,firstks)*gammainc(firstks+1,lambdatrue)/lambdac**firstks)/norm
    else:
        return np.sum(perm(nobs,ks)*gammainc(ks+1,lambdatrue)/lambdac**ks)/norm

def mean(lambdac,nobs):
    ks=np.array(range(nobs+1)).astype(float)
    norm=normalize(lambdac,nobs)
    if nobs>maxk:
        firstks=np.array(range(maxk)).astype(float)
        return np.sum(perm(nobs,firstks)*(firstks+1)/lambdac**firstks)/norm
    else:
        return np.sum(perm(nobs,ks)*(ks+1)/lambdac**ks)/norm

def std(lambdac,nobs):
    ks=np.array(range(nobs+1)).astype(float)
    norm=normalize(lambdac,nobs)
    meani=mean(lambdac,nobs)
    if nobs>maxk:
        firstks=np.array(range(maxk)).astype(float)
        return np.sqrt(np.sum(perm(nobs,firstks)*(firstks+2)*(firstks+1)/lambdac**firstks)/norm-meani**2)
    else:
        return np.sqrt(np.sum(perm(nobs,ks)*(ks+2)*(ks+1)/lambdac**ks)/norm-meani**2)

def percentile(p,lambdac,nobs,exact=False):

    if nobs==0:
        guess=expon.ppf(p/100)
    elif lambdac/nobs>.5:
        guess=expon.ppf(p/100)
    else:
        guess=poisson.ppf(p/100,nobs)
    return fsolve(lambda x:cdf(x,lambdac,nobs)-p/100,guess)[0]

def rvs(lambdac,nobs):
    u=np.random.uniform(0,1)
    return percentile(u*100,lambdac,nobs)

def maxp(lambdac,nobs):
    maxp=nobs-lambdac
    if type(maxp)!=np.ndarray:
        if maxp>0:
            return maxp
        else:
            return 0
    else:
        result=np.array(maxp)
        result[result<0]=0
        
        return result
