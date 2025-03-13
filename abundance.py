from scipy import stats
from scipy.special import gamma,gammaincc
import numpy as np

def norm(ntot,lambdacontam):
    return gammaincc(ntot+1,lambdacontam)

def pmf(ntot,lambdacontam,n):
    pmfi=stats.poisson.pmf(ntot-n,lambdacontam)
    pmfi/=norm(ntot,lambdacontam)
    return pmfi

def cdf(ntot,lambdacontam,n):
    pmfi=pmf(ntot,lambdacontam,np.arange(0,n+1))
    cdfi=np.sum(pmfi)
    return cdfi

def pmf_continuous(ntot,lambdacontam,n):
    pmfc=np.exp(-lambdacontam)*lambdacontam**(ntot-n)/gamma(ntot-n+1)
    pmfc/=norm(ntot,lambdacontam)
    return pmfc

def cdf_continuous(ntot,lambdacontam,n):
    cdfc=1-gammaincc(ntot-n,lambdacontam)/gammaincc(ntot+1,lambdacontam)
    return cdfc

def logpmf(ntot,lambdacontam,n):
    logpmf=stats.poisson.logpmf(ntot-n,lambdacontam)
    logpmf-=np.log(norm(ntot,lambdacontam))
    return logpmf

def logcdf(ntot,lambdacontam,n):
    logpmf=stats.poisson.logcdf(ntot-n,lambdacontam)
    logpmf-=np.log(norm(ntot,lambdacontam))
    return logcdf

def mean(ntot,lambdacontam):
    ntrue=np.arange(0,ntot+1)
    pmfi=pmf(ntot,lambdacontam,ntrue)
    return np.sum(pmfi*ntrue)

def median(ntot,lambdacontam):
    ntrue=np.linspace(0,ntot,num=1000)
    cdfc=cdf_continuous(ntot,lambdacontam,ntrue)
    return np.interp(.5,cdfc,ntrue)

def std(ntot,lambdacontam):
    ntrue=np.arange(0,ntot+1)
    pmfi=pmf(ntot,lambdacontam,ntrue)
    meani=mean(ntot,lambdacontam)
    return np.sqrt(np.sum(pmfi*(ntrue-meani)**2))

def percentile(ntot,lambdacontam,percentile):
    ntrue=np.linspace(0,ntot,num=1000)
    cdfc=cdf_continuous(ntot,lambdacontam,ntrue)
    return np.interp(percentile/100,cdfc,ntrue)

