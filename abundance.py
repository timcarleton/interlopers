from scipy import stats
from scipy.special import gama
import numpy as np

def mean(ntot,lambdacontam):
    ntrue=np.arange(0,ntot)
    pmf=stats.poisson.pmf(ntot-ntrue,lambdacontam)
    return np.sum(pmf*ntrue)

def median(ntot,lambdacontam):
    ntrue=np.linspace(0,ntot,size=1000)
    pmf=np.exp(-lambdacontam)*lambdacontam**(ntot-ntrue)/gamma(ntot-ntrue+1)
    cdf=np.cumsum(pmf)
    return np.interp(.5,ntrue,cdf)

def percentile(ntot,lambdacontam,percentile):
    ntrue=np.linspace(0,ntot,size=1000)
    pmf=np.exp(-lambdacontam)*lambdacontam**(ntot-ntrue)/gamma(ntot-ntrue+1)
    cdf=np.cumsum(pmf)
    return np.interp(percentile/100,ntrue,cdf)
