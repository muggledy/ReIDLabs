from code.viper import get_gog_viper,get_lomo_viper
from code.lomo.xqda import xqda
from code.lomo.tools import mah_dist,calc_cmc,plot_cmc
from code.gog.utils import normalize
from code.tools import shuffle_before_cmc,seek_good_coeffi
import numpy as np
import time

t1=time.time()
probe,gallery=get_gog_viper()

probFeaGOG=normalize(probe.T)
galFeaGOG=normalize(gallery.T)

probe,gallery=get_lomo_viper()
probFeaLOMO=probe.T
galFeaLOMO=gallery.T

numClass = 632
numRank=100
numFolds=10

cFUSION=[]
cGOG=[]
cLOMO=[]

coeffis=[]

for i in range(numFolds):
    p=np.random.permutation(numClass)
    
    half=int(numClass/2)
    probFeaGOG1=probFeaGOG[:,p[:half]]
    galFeaGOG1=galFeaGOG[:,p[:half]]
    WGOG,MGOG,*_=xqda(probFeaGOG1,galFeaGOG1,p[:half],p[:half])

    probFeaLOMO1=probFeaLOMO[:,p[:half]]
    galFeaLOMO1=galFeaLOMO[:,p[:half]]
    WLOMO,MLOMO,*_=xqda(probFeaLOMO1,galFeaLOMO1,p[:half],p[:half])

    probFeaGOG2=probFeaGOG[:,p[half:]]
    galFeaGOG2=galFeaGOG[:,p[half:]]
    distGOG=mah_dist(MGOG,(WGOG.T).dot(probFeaGOG2),(WGOG.T).dot(galFeaGOG2))
    distGOG=distGOG/np.linalg.norm(distGOG,'fro') #see in CMDL
    cGOG.append(calc_cmc(*shuffle_before_cmc(distGOG.T,np.arange(half),np.arange(half)),numRank))

    probFeaLOMO2=probFeaLOMO[:,p[half:]]
    galFeaLOMO2=galFeaLOMO[:,p[half:]]
    distLOMO=mah_dist(MLOMO,(WLOMO.T).dot(probFeaLOMO2),(WLOMO.T).dot(galFeaLOMO2))
    distLOMO=distLOMO/np.linalg.norm(distLOMO,'fro') #
    cLOMO.append(calc_cmc(*shuffle_before_cmc(distLOMO.T,np.arange(half),np.arange(half)),numRank))

    c=calc_cmc(*shuffle_before_cmc((0.75*distGOG+0.25*distLOMO).T,np.arange(half),np.arange(half)),numRank)
    cFUSION.append(c)

    coeffi=seek_good_coeffi([distGOG.T,distLOMO.T],np.arange(half),np.arange(half))
    coeffis.append(coeffi[0])
    print('the best trade-off coeffi:',coeffi)

cFUSION_mean=np.mean(np.array(cFUSION),axis=0)
cGOG_mean=np.mean(np.array(cGOG),axis=0)
cLOMO_mean=np.mean(np.array(cLOMO),axis=0)
print('融合CMC分值：')
print(cFUSION_mean)
print('比较好的系数：')
print(np.mean(np.array(coeffis),axis=0))
print('总的时间消耗：',time.time()-t1)
plot_cmc(np.array([cFUSION_mean,cGOG_mean,cLOMO_mean]),['fusion(viper)','GOG(viper)','LOMO(viper)'],verbose=True)