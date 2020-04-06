from code.viper import get_salience_viper
from code.lomo.tools import calc_cmc,plot_cmc
from code.cspl import cspl
from code.tools import euc_dist
import numpy as np
import time

t1=time.time()
# --Stage1: get salience features--
probFea,galFea=get_salience_viper(pca_n_components=100)

# --Stage2: train and match--
numClass=632
numFlods=10
cs=[]
for i in range(numFlods):
    p=np.random.permutation(numClass)
    half=int(numClass/2)
    probFea1=probFea[:,p[:half]]
    galFea1=galFea[:,p[:half]]
    P1,P2,A=cspl(probFea1,galFea1,iter=200)
    probFea2=probFea[:,p[half:]]
    galFea2=galFea[:,p[half:]]
    dist=euc_dist(P1.dot(probFea2),A.dot(P2.dot(galFea2)))
    c=calc_cmc(dist.T,range(half),range(half),rank=100)
    cs.append(c)
cs=np.array(cs)
c_mean=np.mean(cs,axis=0)
print('all time consumes:',time.time()-t1)
plot_cmc(c_mean,labels=['viper'],verbose=True)
