'''
Rank-1 preserves to be 45% on VIPeR, comparing with paper(49%), 
I don't know if it is my problem or if 49% is not a stable value. 
You can execute ./code/matlab/features/viper_gog_helper.m to get 
official gog descriptors(850s, mine 230s), and rank-1 also be 45%
'''

from initial import *
from zoo.viper import get_gog_viper
from lomo.xqda import xqda
from gog.utils import normalize

t1=time.time()
probe,gallery=get_gog_viper() #mask='ellipse' no help
probFea=normalize(probe.T) #(normalize)seems no great effect
galFea=normalize(gallery.T)

### use features generated by matlab source code
# import scipy.io as sio
# import os
# data=sio.loadmat(os.path.normpath(os.path.join( \
#     os.path.dirname(__file__),'../data/gog_viper.mat')))
# probFea=normalize(data['cam_a'])
# galFea=normalize(data['cam_b'])

numClass=632
numRank=100
numFolds=5

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
print_cmc(c_mean,color=True)
print('time consumes:',time.time()-t1)
plot_cmc(c_mean,['viper'],verbose=True)

'''
probe,gallery=get_gog_viper()
probFea=normalize(probe.T)
galFea=normalize(gallery.T)
feats=np.concatenate((probFea,galFea),axis=1)

from zoo.tools import split_dataset_trials

cs=[]
for trial in split_dataset_trials(list(range(632))*2,[0]*632+[1]*632,'viper',trials=5):
    probFea1=feats[:,trial['indsAtrain']]
    galFea1=feats[:,trial['indsBtrain']]
    W,M,*_=xqda(probFea1,galFea1,trial['labelsAtrain'],trial['labelsBtrain'])
    probFea2=feats[:,trial['indsAtest']]
    galFea2=feats[:,trial['indsBtest']]
    dist=mah_dist(M,(W.T).dot(probFea2),(W.T).dot(galFea2))
    c=calc_cmc(dist.T,trial['labelsAtest'],trial['labelsBtest'],100)
    cs.append(c)

c_mean=np.mean(np.array(cs),axis=0)
print('CMC:')
print_cmc(c_mean,color=True)
plot_cmc(c_mean,['viper'],verbose=True)
'''