import numpy as np
import os.path
from .lomo.lomo import load_data,get_hsv,get_siltp
from .lomo.tools import getcwd
from .salience.extractDescriptors import extractDescriptorsFromCam
from .gog.gog import GOG
from .gog.set_parameter import get_default_parameter
from sklearn.decomposition import PCA

cwd=getcwd(__file__)
cam_a_dir=os.path.normpath(os.path.join(cwd,'../../images/VIPeR.v1.0/cam_a'))
cam_b_dir=os.path.normpath(os.path.join(cwd,'../../images/VIPeR.v1.0/cam_b'))

def pca_reduct_dim(probFea,galFea,n_components,W_file=None):
    '''input probe(n,d) and gallery(n,d), output probe(d',n) and gallery(d',n) 
       after dimension reduction'''
    print('probe features(before pca):%s, gallery features(before pca):%s'% \
                                    (probFea.shape[::-1],galFea.shape[::-1]))
    if W_file==None or (not os.path.exists(W_file)):
        pca=PCA(n_components=n_components)
        probFea=pca.fit_transform(probFea)
        probEnergy=sum(pca.explained_variance_ratio_)
        probWT=pca.components_
        galFea=pca.fit_transform(galFea)
        galEnergy=sum(pca.explained_variance_ratio_)
        galWT=pca.components_
        probFea=probFea.T
        galFea=galFea.T
        np.savez(W_file,probWT=probWT,probEnergy=probEnergy,galWT=galWT,galEnergy=galEnergy)
    else:
        print('get pca\'s project matrix(n_components=%d) from local'%n_components)
        data=np.load(W_file)
        probWT,probEnergy=data['probWT'],data['probEnergy']
        probFea=probWT.dot(probFea.T)
        galWT,galEnergy=data['galWT'],data['galEnergy']
        galFea=galWT.dot(galFea.T)
    print( \
        'reduct probe dimension(energy:%.2f%%):%s, reduct gallery dimension(energy:%.2f%%):%s' \
                          %(probEnergy,probFea.shape,galEnergy,galFea.shape))
    return probFea,galFea

def get_lomo_viper(pca_n_components=None):
    '''return probe(632*26960,cam_a) and gallery(632*26960,cam_b) of VIPeR dataset's 
       LOMO descriptors, if pca_n_components!=None, e.g. 100, func will return probe
       (100,632) and gallery(100,632) after PCA'''
    feat_file=os.path.join(cwd,'../../data/lomo_features_viper.npz')
    if not os.path.exists(feat_file):
        imgs=load_data(cam_a_dir)
        fea1=get_hsv(imgs)
        fea2=get_siltp(imgs,R=3)
        fea3=get_siltp(imgs,R=5)
        probe=np.hstack((fea1,fea2,fea3))
        imgs=load_data(cam_b_dir)
        fea1=get_hsv(imgs)
        fea2=get_siltp(imgs,R=3)
        fea3=get_siltp(imgs,R=5)
        gallery=np.hstack((fea1,fea2,fea3))
        np.savez(feat_file,probe=probe,gallery=gallery)
    else:
        print('viper lomo features have existed!')
        data=np.load(feat_file)
        probe,gallery=data['probe'],data['gallery']
    if pca_n_components!=None:
        W_file=os.path.join(cwd,'../../data/lomo_features_viper_pca%d.npz'%pca_n_components)
        return pca_reduct_dim(probe,gallery,pca_n_components,W_file)
    return probe,gallery

def get_salience_viper(pca_n_components=None):
    '''return probe(632*201600,cam_a) and gallery(632*201600,cam_b) of VIPeR's 
       salience descriptors at image-level, you can set pca_n_components like 
       get_lomo_viper'''
    feat_file=os.path.join(cwd,'../../data/salience_features_viper.npz')
    if not os.path.exists(feat_file):
        probe=extractDescriptorsFromCam(cam_a_dir)[0].T
        gallery=extractDescriptorsFromCam(cam_b_dir)[0].T
        np.savez(feat_file,probe=probe,gallery=gallery)
    else:
        print('viper salience features have existed!')
        data=np.load(feat_file)
        probe,gallery=data['probe'],data['gallery']
    if pca_n_components!=None:
        W_file=os.path.join(cwd,'../../data/salience_features_viper_pca%d.npz'%pca_n_components)
        return pca_reduct_dim(probe,gallery,pca_n_components,W_file)
    return probe,gallery

def get_gog_viper():
    '''return probe(n*dim,cam_a) and gallery(n*dim,cam_b) of VIPeR dataset's GOG 
       descriptors'''
    feat_file=os.path.join(cwd,'../../data/gog_features_viper.npz')
    if not os.path.exists(feat_file):
        dim=0
        param=get_default_parameter()
        for i in range(4): #get fusion(four color space:RGB,HSV,Lab,nRnG) GOG 
                           #descriptor's dim(equal for arbitrary image size)
            param.lfparam.lf_type=i
            dim+=param.dimension
        imgs=load_data(cam_a_dir) #probe
        n=imgs.shape[-1]
        probe=np.zeros((n,dim))
        for i in range(n):
            im=imgs[...,i]
            feas=[]
            for j in range(4):
                param=get_default_parameter(j)
                feas.append(GOG(im,param))
            probe[i,:]=np.hstack(feas)
        imgs=load_data(cam_b_dir) #gallery
        n=imgs.shape[-1]
        gallery=np.zeros((n,dim))
        for i in range(n):
            im=imgs[...,i]
            feas=[]
            for j in range(4):
                param=get_default_parameter(j)
                feas.append(GOG(im,param))
            gallery[i,:]=np.hstack(feas)
        np.savez(feat_file,probe=probe,gallery=gallery)
    else:
        print('viper gog features have existed!')
        data=np.load(feat_file)
        probe,gallery=data['probe'],data['gallery']
    return probe,gallery