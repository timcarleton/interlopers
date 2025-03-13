from scipy import stats
from scipy.special import gamma
import numpy as np

def mean(ntot,lambdacontam):
    ntrue=np.arange(0,ntot)
    pmf=stats.poisson.pmf(ntot-ntrue,lambdacontam)
    return np.sum(pmf*ntrue)

def median(ntot,lambdacontam):
    ntrue=np.linspace(0,ntot,num=1000)
    pmf=np.exp(-lambdacontam)*lambdacontam**(ntot-ntrue)/gamma(ntot-ntrue+1)
    cdf=np.cumsum(pmf)
    return np.interp(.5,ntrue,cdf)

def std(ntot,lambdacontam):
    ntrue=np.arange(0,ntot)
    pmf=stats.poisson.pmf(ntot-ntrue,lambdacontam)
    return np.sqrt(np.sum(pmf*(ntrue-mean(ntot,lambdacontam))**2))

def percentile(ntot,lambdacontam,percentile):
    ntrue=np.linspace(0,ntot,num=1000)
    pmf=np.exp(-lambdacontam)*lambdacontam**(ntot-ntrue)/gamma(ntot-ntrue+1)
    cdf=np.cumsum(pmf)
    return np.interp(percentile/100,ntrue,cdf)

def pmf(ntot,lambdacontam,n):
    pmf=stats.poisson.pmf(ntot-n,lambdacontam)
    return pmf

def pmf_continuous(ntot,lambdacontam,n):
    pmf=np.exp(-lambdacontam)*lambdacontam**(ntot-n)/gamma(ntot-n+1)
    return pmf

def cdf(ntot,lambdacontam,n):
    cdf=stats.poisson.cdf(ntot-n,lambdacontam)
    return cdf

def logpmf(ntot,lambdacontam,n):
    logpmf=stats.poisson.logpmf(ntot-n,lambdacontam)
    return logpmf

def logcdf(ntot,lambdacontam,n):
    logpmf=stats.poisson.logcdf(ntot-n,lambdacontam)
    return logcdf
