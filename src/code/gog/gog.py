'''
References:
[1] Matsukawa T , Okabe T , Suzuki E , et al. Hierarchical Gaussian Descriptor for 
    Person Re-identification[C]// 2016 IEEE Conference on Computer Vision and Pattern 
    Recognition (CVPR). IEEE, 2016.
muggledy 2020/4/3
'''

import numpy as np
import logging
from .get_pixel_features import get_pixel_features
from .utils import get_patches,window_nd
from .set_parameter import get_default_parameter
from .gauss import patch_gaussian

logging.basicConfig(level=logging.INFO,format=
                                 '[%(asctime)s] %(name)s(%(levelname)s): %(message)s',
                                                #datefmt='%a %d %b %Y %H:%M:%S'
                                                )
logger = logging.getLogger(__name__) #https://zhuanlan.zhihu.com/p/38782314

def split_strips(X,n):
    '''split X(h,w,c) into n horizontal strips, func will return a ndarray of shape
       (n,roi_h,roi_w,c)'''
    t={1:1,2:3/4,3:1/2,4:1/2,5:1/3,6:1/4,7:1/4,8:7/32,9:1/6,10:3/20} #n should be 1~10
    h,w,_=X.shape
    roi_h=int(h*t[n])
    roi_w=w
    steps_h=roi_h-(np.ceil((roi_h*n-h)/(n-1))) if n!=1 else roi_h
    steps_w=roi_w
    ret=get_patches(X,(roi_h,roi_w),(steps_h,steps_w))
    if n==1:
        return np.expand_dims(ret,0)
    return ret

def GOG(img,param=None):
    '''get Gaussian of Gaussian(GOG) descriptor of img(h,w,3), param is the parameters 
       required for running, you can get param from parameter.get_default_parameter() 
       quickly'''
    if param==None:
        param=get_default_parameter()
    logger.info('get %s descriptor of the image'%param.name)
    F=get_pixel_features(img,param.lfparam)
    regions=split_strips(F,param.G)
    logger.info('split into %d horizontal strips%s'%(param.G,str(regions.shape)))
    spatches=window_nd(regions,param.k,param.p,(1,2))
    spatches=np.rollaxis(spatches,2)
    logger.info( \
      'split into fine-grained patches(with size %d, interval %d) for each strip%s'% \
                                           (param.k,param.p,str(spatches.shape)))
    t=patch_gaussian(spatches,None,param.epsilon0)
    if param.ifweight:
        logger.info('use weight on region gaussian')
        img_x=np.broadcast_to(np.arange(1,img.shape[1]+1)[None,:,None],(*(img.shape[:-1]),1))
        img_x=split_strips(img_x,param.G)
        img_x=window_nd(img_x,param.k,param.p,(1,2))
        img_x=np.rollaxis(img_x,2)
        weights=img_x[...,img_x.shape[-2]//2,img_x.shape[-1]//2]
        weights=np.exp((-(weights-(img.shape[1]/2))**2)/(2*(img.shape[1]/4)**2))
    else:
        weights=None
    t=patch_gaussian(t,weights,param.epsilon0).ravel() #region gaussian(see regions 
                                                #in img like patches in region)
    return t
