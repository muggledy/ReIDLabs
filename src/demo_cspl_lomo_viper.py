from code.viper import get_lomo_viper
from code.lomo.tools import getcwd,calc_cmc,plot_cmc
from code.cspl import cspl
from code.tools import euc_dist
from sklearn.decomposition import PCA
import os.path
import numpy as np
import time

cwd=getcwd(__file__)

t1=time.time()
# --Stage1: get lomo features--
probFea,galFea=get_lomo_viper() #mask='ellipse' no help, see note 2
'''use official lomo features(with retinex), compared to mine, no difference
import scipy.io as sio
data=sio.loadmat(os.path.normpath(os.path.join(cwd,'../data/viper_lomo.mat')))['descriptors']
probFea=data[:,:632].T
galFea=data[:,632:].T
'''
print('load probe features:%s, load gallery features:%s'%(probFea.shape[::-1],galFea.shape[::-1]))
pca=PCA(n_components=100) #set to 600(99%) and set to 100(56%), no difference. PCA refer: https://www.cnblogs.com/pinard/p/6243025.html
probFea=pca.fit_transform(probFea)
probEnergy=sum(pca.explained_variance_ratio_)
galFea=pca.fit_transform(galFea)
galEnergy=sum(pca.explained_variance_ratio_)
probFea=probFea.T
galFea=galFea.T
print('reduct probe dimension(energy:%.2f%%):%s, reduct gallery dimension(energy:%.2f%%):%s'%(probEnergy,probFea.shape,galEnergy,galFea.shape))

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

'''note 1: the PCA reduction of above code may not be reasonable, but the results are same
from code.viper import get_lomo_viper
from code.lomo.tools import getcwd,calc_cmc,plot_cmc
from code.cspl import cspl
from code.tools import euc_dist
from sklearn.decomposition import PCA
import os.path
import numpy as np
import time

cwd=getcwd(__file__)

t1=time.time()
# --Stage1: get lomo features--
probFea,galFea=get_lomo_viper(mask='ellipse')
probFea=probFea.T
galFea=galFea.T

# --Stage2: train and match--
numClass=632
numFlods=10
cs=[]
for i in range(numFlods):
    p=np.random.permutation(numClass)
    half=int(numClass/2)
    probFea1=probFea[:,p[:half]]
    galFea1=galFea[:,p[:half]]

    pca1=PCA(n_components=315)
    probFea1=pca1.fit_transform(probFea1.T).T
    probEnergy=sum(pca1.explained_variance_ratio_)
    
    pca2=PCA(n_components=315)
    galFea1=pca2.fit_transform(galFea1.T).T
    galEnergy=sum(pca2.explained_variance_ratio_)

    print('reduct probe dimension(energy:%.2f%%):%s, reduct gallery dimension(energy:%.2f%%):%s'%(probEnergy,probFea1.shape,galEnergy,galFea1.shape))

    P1,P2,A=cspl(probFea1,galFea1,iter=200,convergence=True)
    probFea2=probFea[:,p[half:]]
    galFea2=galFea[:,p[half:]]

    probFea2=pca1.transform(probFea2.T).T
    galFea2=pca2.transform(galFea2.T).T

    dist=euc_dist(P1.dot(probFea2),A.dot(P2.dot(galFea2)))
    c=calc_cmc(dist.T,range(half),range(half),rank=100)
    cs.append(c)
cs=np.array(cs)
c_mean=np.mean(cs,axis=0)
print('all time consumes:',time.time()-t1)
plot_cmc(c_mean,labels=['viper'],verbose=True)
'''
'''note 2
from code.tools import create_ellipse_mask
import os.path
import cv2
import matplotlib.pyplot as plt
from code.lomo.lomo import load_data
import numpy as np

img=os.path.join(os.path.dirname(__file__),'../images/VIPeR.v1.0/cam_a/')
imgs=load_data(img,'ellipse')
imgs=imgs[...,np.random.randint(0,631,1)]
plt.imshow(imgs[...,0][...,::-1])
plt.show()
'''