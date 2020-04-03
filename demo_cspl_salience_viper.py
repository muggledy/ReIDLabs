from code.salience.extractDescriptors import extractDescriptorsFromCam
from code.lomo.tools import getcwd,calc_cmc,plot_cmc
from code.cspl import cspl
from code.tools import euc_dist
from sklearn.decomposition import PCA
import os.path
import numpy as np
import time

cwd=getcwd(__file__)

t1=time.time()
# --Stage1: get salience features--
feat_file=os.path.join(cwd,'data/salience_features_viper.npz')

if not os.path.exists(feat_file):
    img_path=os.path.normpath(os.path.join(cwd,'images/VIPeR.v1.0/cam_a'))
    probe=extractDescriptorsFromCam(img_path)[0]
    img_path=os.path.normpath(os.path.join(cwd,'images/VIPeR.v1.0/cam_b'))
    gallery=extractDescriptorsFromCam(img_path)[0]
    np.savez(feat_file,probe=probe.T,gallery=gallery.T)
else:
    print('viper salience features have existed!')

data=np.load(feat_file)
probFea=data['probe']
galFea=data['gallery']
print('load probe features:%s, load gallery features:%s'%(probFea.shape[::-1],galFea.shape[::-1]))

n_components=100
W_file=os.path.join(cwd,'data/salience_features_viper_pca%d.npz'%n_components)
if not os.path.exists(W_file):
    pca=PCA(n_components=n_components)
    probFea=pca.fit_transform(probFea)
    probW=pca.components_
    probEnergy=sum(pca.explained_variance_ratio_)
    galFea=pca.fit_transform(galFea)
    galW=pca.components_
    galEnergy=sum(pca.explained_variance_ratio_)
    np.savez(W_file,probW=probW,probEnergy=probEnergy,galW=galW,galEnergy=galEnergy)
    probFea=probFea.T
    galFea=galFea.T
else:
    print('get pca%d from local'%n_components)
    data=np.load(W_file)
    probW=data['probW']
    probEnergy=data['probEnergy']
    probFea=probW.dot(probFea.T)
    galW=data['galW']
    galEnergy=data['galEnergy']
    galFea=galW.dot(galFea.T)
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
    P1,P2,A=cspl(probFea1,galFea1,iter=150)
    probFea2=probFea[:,p[half:]]
    galFea2=galFea[:,p[half:]]
    dist=euc_dist(P1.dot(probFea2),A.dot(P2.dot(galFea2)))
    c=calc_cmc(dist.T,range(half),range(half),rank=100)
    cs.append(c)
cs=np.array(cs)
c_mean=np.mean(cs,axis=0)
print('all time consumes:',time.time()-t1)
plot_cmc(c_mean,labels=['viper'],verbose=True)
