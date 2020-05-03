from .logm import logm0,logm1
from .utils import matrix_regularize
import numpy as np
import logging
logger = logging.getLogger(__name__)

def calc_mean_conv(X,weights=None):
    '''X's last three dims represent one patch(h,w,d), we calculate these patches' 
       covariance matrix and mean vector, if X's shape is (...,h,w,d), conv will be 
       (...,d,d) and mean be (...,d). Note that weights' shape should be (...,h,w)'''
    flag=isinstance(weights,np.ndarray)
    if flag:
        if weights.shape!=X.shape[:-1]:
            raise ValueError('weights\' shape should be (...,h,w)!')
        ws=np.sum(weights,axis=(-2,-1))
        mean=np.sum(X*weights[...,None],axis=(-3,-2))/(ws[...,None])
    else:
        mean=np.mean(X,axis=(-3,-2))
    norm_X=X-mean[...,None,None,:]
    norm_X=norm_X.reshape(*(norm_X.shape[:-3]),-1,norm_X.shape[-1])
    norm_XT=np.swapaxes(norm_X,-2,-1)
    if flag:
        conv=np.matmul(norm_XT*weights.reshape( \
             *(weights.shape[:-2]),-1)[...,None,:],norm_X)/(ws[...,None,None])
    else:
        conv=np.matmul(norm_XT,norm_X)/(norm_X.shape[-2]-1)
    return mean,conv

def half_vec(X):
    '''X may be a N-d array, we half-vectoralize X's last two dimensions, note that 
       these sub-matrix should be symmetric, if X's shape is (...,d,d), we will get a 
       matrix of (...,(d^2+3d)/2+1)'''
    d,_=X.shape[-2:]
    if d!=_:
        raise ValueError('X\'s last two dimensions must be equal!')
    indx,indy=np.triu_indices(d)
    return X[...,indx,indy]

def patch_gaussian(X,weights=None,eps=0.001,**kwargs):
    '''GoG paper refers Integral Images for Fast Covariance Computation in "Pedestrian 
       Detection via Classification on Riemannian Manifolds", utilizing multi thread 
       with matlab, i don't want to follow in python because it may be extremely poor 
       , so i use matrix parallel computing of numpy herein, despite of much repeated 
       calculations, it is still very fast. See func calc_mean_conv about X, and 
       weights's effect is exerted on patches' mean and conv. This function can also be 
       utilized by region gaussian, don't be confused by the func name'''
    d=X.shape[-1]
    stage_name=kwargs.get('stage','patch/region')
    logger.info('calc %s gaussian vectors%s'%(stage_name, \
                                        str((*X.shape[:-3],(d**2+3*d)//2+1)))) #equal 
                                        #to the shape of half_vec(sP) at last
    if len(X.shape)<3:
        raise ValueError('X\'s shape should be (...,h,w,c)!')
    mean,conv=calc_mean_conv(X,weights)
    mmT=np.matmul(mean[...,None],mean[...,None,:])
    sP=np.zeros((*(mmT.shape[:-2]),*((d+1,)*2)))
    sP[...,:-1,:-1]=conv+mmT
    sP[...,:-1,[-1]]=mean[...,None]
    sP[...,[-1],:-1]=mean[...,None,:]
    sP[...,-1,-1]=1
    if stage_name=='patch':
        #eps*=np.maximum(np.trace(conv,axis1=-1,axis2=-2),0.01) #trace norm regularization
        pass
    elif stage_name=='region':
        #eps*=np.trace(conv,axis1=-1,axis2=-2) #seems no effect or no improvement?
        pass
    dets=(np.linalg.det(matrix_regularize(conv,eps))**(-1/(d+1)))[...,None,None]
    sP*=dets #patch gaussian matrix
    logger.info('calc logm of all %s gaussian...'%stage_name)
    sP=logm1(sP) #see https://ww2.mathworks.cn/help/matlab/ref/logm.html and
    #https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.linalg.logm.html
    diags=sP[...,range(d+1),range(d+1)]
    sP*=np.sqrt(2)
    sP[...,range(d+1),range(d+1)]=diags
    logger.info('half-vectoralize %s gaussian'%stage_name)
    return half_vec(sP)
