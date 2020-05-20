'''
Rank1 preserves to be 45% on VIPeR with official arguments in 
paper(49%), i don't know if it is my problem or if 49% is a 
unstable value
'''

from code.viper import get_gog_viper
from code.lomo.xqda import xqda
from code.lomo.tools import mah_dist,calc_cmc,plot_cmc
from code.gog.utils import normalize
import numpy as np
import time

t1=time.time()
probe,gallery=get_gog_viper() #mask='ellipse' no help

probFea=normalize(probe.T) #seems no great effect
galFea=normalize(gallery.T)

#probFea=probe.T
#galFea=gallery.T

numClass = 632
numRank=100
numFolds=10

cs=np.zeros((numFolds,numRank),np.float)

for i in range(numFolds):
    p=np.random.permutation(numClass)
    
    half=int(numClass/2)
    probFea1=probFea[:,p[:half]]
    galFea1=galFea[:,p[:half]]
    W,M,*_=xqda(probFea1,galFea1,p[:half],p[:half])
    probFea2=probFea[:,p[half:]]
    galFea2=galFea[:,p[half:]]
    dist=mah_dist(M,(W.T).dot(probFea2),(W.T).dot(galFea2))
    c=calc_cmc(dist.T,np.arange(half),np.arange(half),numRank)
    cs[i]=c

c_mean=np.mean(cs,axis=0)
print('CMC:')
print(c_mean)
print('time consumes:',time.time()-t1)
plot_cmc(c_mean,['viper'],verbose=True)