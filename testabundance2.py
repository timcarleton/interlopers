import numpy as np
from scipy.stats import poisson
import twopoisson
import matplotlib.pyplot as plt
plt.style.use('../python/stylesheet.txt')

lambdacont=10

lambdareal=np.random.uniform(0,20,size=100000)
populations=[]

for i in range(500):
    populationsi=[]
    for lreal in range(len(lambdareal)):
        populationsi.append(np.random.poisson(lambdacont)+np.random.poisson(lambdareal[lreal]))
    populations.append(populationsi)

populations=np.array(populations)

plt.clf()
for ntest in np.arange(10,21,2):
    w=np.where(populations==ntest)[1]

    hst=np.histogram(lambdareal[w],density=1,bins=50,range=[0,20])
    bins=hst[1]
    midbins=(bins[0:-1]+bins[1:])/2
    pred=np.array([twopoisson.cdf(bins[i+1],lambdacont,ntest)-twopoisson.cdf(bins[i],lambdacont,ntest) for i in range(len(bins)-1)])

    #plt.plot(midbins,hst[0],color=plt.get_cmap('plasma')(ntest/5),ls='--')
    plt.plot(midbins,hst[0],color=plt.get_cmap('plasma')((ntest-10)/20),ls='--')
    plt.plot(midbins,poisson.pmf(np.array(midbins).astype(int),(ntest-lambdacont)),color=plt.get_cmap('plasma')((ntest-10)/20),ls=':')
    plt.plot(midbins,pred/(bins[1]-bins[0]),color=plt.get_cmap('plasma')((ntest-10)/20),label=r'$N_{\rm tot}=$'+str(ntest))

plt.xlabel(r'$\lambda_{\rm true}$')
plt.ylabel(r'$dP(\lambda_{\rm true})/d\lambda_{\rm true}$')
plt.legend()
plt.xlim(0,20)
plt.yscale('log')
plt.ylim(1E-3,1)
plt.title(r'$\lambda_{\rm contam}=10$',y=.93)
plt.savefig('testinf10.png')
    
