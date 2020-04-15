import numpy as np
import cv2
from .utils import to_ycbcr
import logging
logger = logging.getLogger(__name__)
#from ..lomo.tools import measure_time

def get_gradmap(X,binnum):
    '''X is the input single-channel image(h,w), binnum is the number of graient 
       orientation bins. Func will return qori(quantized gradient orientation map
       (soft voting) with size (h,w,binnum)) and ori(gradient orientations) and 
       mag(gradient magnitude)'''
    X=X.astype('double')
    hx=np.array([-1,0,1])[None,:]
    hy=(-hx).T
    grad_x=cv2.filter2D(X,-1,hx)
    grad_y=cv2.filter2D(X,-1,hy)
    ori=(np.arctan2(grad_x,grad_y)+np.pi)*180/np.pi
    mag=np.sqrt(grad_x**2+grad_y**2)
    binwidth=360/binnum
    
    IND=np.floor(ori/binwidth).astype('int')
    ref1=IND*binwidth
    ref2=(IND+1)*binwidth
    dif1=ori-ref1
    dif2=ref2-ori
    weight1=dif2/(dif1+dif2)
    weight2=dif1/(dif1+dif2)
    h,w=X.shape
    qori=np.zeros((h,w,binnum))
    IND[IND==binnum]=0
    IND1=IND
    IND2=IND+1
    IND2[IND2==binnum]=0
    
    x,y=np.meshgrid(range(w),range(h))
    qori[y.ravel(),x.ravel(),IND1.ravel()]=weight1.ravel()
    qori[y.ravel(),x.ravel(),IND2.ravel()]=weight2.ravel()
    
    return qori,ori,mag

def get_pixel_features(img,lfparam):
    '''extract pixel features map defined by lfparam from input BGR img(h,w,3). Func 
       returns a pixel feature map(h,w,lfparam.num_element), a matrix'''
    h,w=img.shape[:2]
    ret=np.zeros((h,w,lfparam.num_element),dtype='double')
    logger.info('get gog\'s pixel feature map%s'%str(ret.shape))
    t=np.arange(1,h+1)
    t=np.broadcast_to(t,(w,h)).T.astype('double')
    curdimpos=0
    ret[...,curdimpos]=t/h #y
    curdimpos+=1
    
    binnum=lfparam.lf_bins
    img_ycbcr=to_ycbcr(img).astype('double')
    img_ycbcr[...,0]=(img_ycbcr[...,0]-16)/235
    qori,_,mag=get_gradmap(img_ycbcr[...,0],binnum)
    Yq=qori*mag[...,None]
    ret[...,curdimpos:curdimpos+binnum]=Yq #M_theta
    curdimpos+=binnum
    
    if lfparam.lf_type==0: #RGB
        ret[...,curdimpos:curdimpos+3]=img.astype('double')[...,::-1]/255
    elif lfparam.lf_type==1: #Lab
        img_lab=cv2.cvtColor(img,cv2.COLOR_BGR2Lab).astype('double')
        img_lab[...,0]=img_lab[...,0]/100
        img_lab[...,[1,2]]=(img_lab[...,[1,2]]+50)/100
        ret[...,curdimpos:curdimpos+3]=img_lab
    elif lfparam.lf_type==2: #HSV
        ret[...,curdimpos:curdimpos+3]=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype('double')/255
    elif lfparam.lf_type==3: #nRnG
        sumVal=np.sum(img,axis=2)
        sumVal[sumVal<1]=1
        ret[...,curdimpos:curdimpos+2]=img[...,[0,1]].astype('double')/sumVal[...,None]
    return ret
