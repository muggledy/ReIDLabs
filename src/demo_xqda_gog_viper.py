from code.viper import get_gog_viper
from code.lomo.xqda import xqda
from code.lomo.tools import mah_dist,calc_cmc,plot_cmc
import numpy as np
import time

t1=time.time()
probe,gallery=get_gog_viper()

numClass = 632
probFea=probe.T
galFea=gallery.T

numRank=100
numFolds=1
k=np.random.randint(0,numFolds)
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
print('CMC分值：')
print(c_mean)
print('总的时间消耗：',time.time()-t1)
plot_cmc(c_mean,['viper'],verbose=True)