from code.lomo.lomo import load_data,get_hsv,get_siltp
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
feat_file=os.path.join(cwd,'data/lomo_features_viper.npz')

if not os.path.exists(feat_file):
    img_path=os.path.normpath(os.path.join(cwd,'images/VIPeR.v1.0/cam_a'))
    imgs=load_data(img_path)
    fea1=get_hsv(imgs)
    fea2=get_siltp(imgs,R=3)
    fea3=get_siltp(imgs,R=5)
    probe=np.hstack((fea1,fea2,fea3))
    img_path=os.path.normpath(os.path.join(cwd,'images/VIPeR.v1.0/cam_b'))
    imgs=load_data(img_path)
    fea1=get_hsv(imgs)
    fea2=get_siltp(imgs,R=3)
    fea3=get_siltp(imgs,R=5)
    gallery=np.hstack((fea1,fea2,fea3))
    np.savez(feat_file,probe=probe,gallery=gallery)
else:
    print('viper lomo features have existed!')

#'''使用我写的lomo特征
data=np.load(feat_file)
probFea=data['probe']
galFea=data['gallery']
#'''
'''使用官方带有retinex的lomo特征，没区别
import scipy.io as sio
data=sio.loadmat(os.path.normpath(os.path.join(cwd,'data/viper_lomo.mat')))['descriptors']
probFea=data[:,:632].T
galFea=data[:,632:].T
'''
print('load probe features:%s, load gallery features:%s'%(probFea.shape[::-1],galFea.shape[::-1]))
pca=PCA(n_components=100) #设为600(99%)和设为100(56%)没区别，惊奇。PCA参考：https://www.cnblogs.com/pinard/p/6243025.html
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
