'''
This experiment indicates that LOMO and GOG are not complement, CMDL has proved that 
multi-level(patch-level+image-level) features fusion will bring some big improvement 
in spite of its' error in part-level. I mean two image-level fusion may not be good, 
just maybe. I added salience later with the same conclusion
'''

from code.viper import get_gog_viper,get_lomo_viper,get_salience_viper
from code.lomo.xqda import xqda
from code.lomo.tools import mah_dist,calc_cmc,plot_cmc
from code.gog.utils import normalize
from code.tools import seek_good_coeffi
import numpy as np
import time

t1=time.time()
probe,gallery=get_gog_viper()

probFeaGOG=normalize(probe.T)
galFeaGOG=normalize(gallery.T)

probe,gallery=get_lomo_viper()
probFeaLOMO=probe.T
galFeaLOMO=gallery.T

probe,gallery=get_salience_viper()
probFeaSALIENCE=probe.T
galFeaSALIENCE=gallery.T

numClass = 632
numRank=100
numFolds=2

cFUSION,cGOG,cLOMO,cSALIENCE=[],[],[],[]

#coeffis=[] #uncomment directly to seek good trade-off coefficient

for i in range(numFolds):
    p=np.random.permutation(numClass)
    half=int(numClass/2)

    probFeaGOG1=probFeaGOG[:,p[:half]]
    galFeaGOG1=galFeaGOG[:,p[:half]]
    WGOG,MGOG,*_=xqda(probFeaGOG1,galFeaGOG1,p[:half],p[:half])

    probFeaLOMO1=probFeaLOMO[:,p[:half]]
    galFeaLOMO1=galFeaLOMO[:,p[:half]]
    WLOMO,MLOMO,*_=xqda(probFeaLOMO1,galFeaLOMO1,p[:half],p[:half])

    probFeaSALIENCE1=probFeaSALIENCE[:,p[:half]]
    galFeaSALIENCE1=galFeaSALIENCE[:,p[:half]]
    WSALIENCE,MSALIENCE,*_=xqda(probFeaSALIENCE1,galFeaSALIENCE1,p[:half],p[:half])

    probFeaGOG2=probFeaGOG[:,p[half:]]
    galFeaGOG2=galFeaGOG[:,p[half:]]
    distGOG=mah_dist(MGOG,(WGOG.T).dot(probFeaGOG2),(WGOG.T).dot(galFeaGOG2))
    distGOG=distGOG/np.linalg.norm(distGOG,'fro') #normalize, see in CMDL
    cGOG.append(calc_cmc(distGOG.T,np.arange(half),np.arange(half),numRank))

    probFeaLOMO2=probFeaLOMO[:,p[half:]]
    galFeaLOMO2=galFeaLOMO[:,p[half:]]
    distLOMO=mah_dist(MLOMO,(WLOMO.T).dot(probFeaLOMO2),(WLOMO.T).dot(galFeaLOMO2))
    distLOMO=distLOMO/np.linalg.norm(distLOMO,'fro')
    cLOMO.append(calc_cmc(distLOMO.T,np.arange(half),np.arange(half),numRank))

    probFeaSALIENCE2=probFeaSALIENCE[:,p[half:]]
    galFeaSALIENCE2=galFeaSALIENCE[:,p[half:]]
    distSALIENCE=mah_dist(MSALIENCE,(WSALIENCE.T).dot(probFeaSALIENCE2), \
        (WSALIENCE.T).dot(galFeaSALIENCE2))
    distSALIENCE=distSALIENCE/np.linalg.norm(distSALIENCE,'fro')
    cSALIENCE.append(calc_cmc(distSALIENCE.T,np.arange(half),np.arange(half),numRank))

    c=calc_cmc((0.55*distGOG+0.3*distLOMO+0.15*distSALIENCE).T,np.arange(half), \
        np.arange(half),numRank)
    cFUSION.append(c)

    #coeffi=seek_good_coeffi([distGOG.T,distLOMO.T,distSALIENCE.T],np.arange(half),np.arange(half))
    #coeffis.append(coeffi[0])
    #print('the best trade-off coeffi:',coeffi)

cFUSION_mean=np.mean(np.array(cFUSION),axis=0)
cGOG_mean=np.mean(np.array(cGOG),axis=0)
cLOMO_mean=np.mean(np.array(cLOMO),axis=0)
cSALIENCE_mean=np.mean(np.array(cSALIENCE),axis=0)
print('CMC(fusion):')
print(cFUSION_mean)
#print('good fusion coefficients:')
#print(np.mean(np.array(coeffis),axis=0))
print('time consumes:',time.time()-t1)
plot_cmc(np.array([cFUSION_mean,cGOG_mean,cLOMO_mean,cSALIENCE_mean]), \
    ['fusion(viper)','GOG(viper)','LOMO(viper)','SALIENCE(viper)'],verbose=True)