'''
2019/11/23
'''

import numpy as np
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from lomo.tools import normc #由于有许多同名的tools文件，如果from tools import normc可能会发生导入错误
import cv2
from lomo.siltp import calc_siltp2
from zoo.tools import create_ellipse_mask
#from .retinex.retinex import retinex_gimp as retinex

def load_data(file_path,mask=None):
    '''load images in file_path into 4-D ndarray(h,w,3,n)'''
    txls=[i for i in os.listdir(file_path) if i.endswith(('.jpg','.png','.bmp'))]
    n=len(txls)
    im0=cv2.imread(os.path.join(file_path,txls[0]))
    if mask=='ellipse':
        ellipse_mask=create_ellipse_mask(im0.shape[:-1])
        im0=cv2.bitwise_and(im0,im0,mask=ellipse_mask)
    imgs=np.zeros((*im0.shape,n),dtype='uint8')
    imgs[...,0]=im0
    for i,tx in enumerate(txls[1:],1):
        tx=os.path.join(file_path,tx)
        img=cv2.imread(tx)
        if mask=='ellipse':
            img=cv2.bitwise_and(img,img,mask=ellipse_mask)
        imgs[...,i]=img
    print('load images into 4-D: ',imgs.shape)
    return imgs

def slide_wins(a,winsize,steps=(1,1),flag=1):
    '''
    parameters:
        a: a panel(2D array) for window sliding, in fact it can be 3D, you can see it as a container of some 
                panels, the 0 axis length of a is the number of panels, i do the same operation on these 
                panels just like on the 2D array a.
        winsize: a tuple of the sliding window's height and width.
        steps: a tuple of the sliding strides in vertical and horizontal directions. Default: (1,1).
        flag: 0 or 1, please see return value [wins]. Default: 1.
    returns:
        wins: a 2D array contains all sub windows, each row represents a subwin. Note that the sliding 
                window moves from left to right and then from top to bottom by default value of flag
                , if you want to move from top to bottom firstly and then from left to right, set flag 
                to 0. If a is 3D, wins should be 3D too, each layer represents the subwins of the corresponding 
                layer in a.
        (rows,cols): the row index and column index indicates the left-top position coordinate of all sub windows.
    examples:
        slide_wins(np.arange(24).reshape(4,-1),(2,4),(2,2))
    '''
    if len(a.shape)==3:
        ah,aw=a[0].shape
    elif len(a.shape)==2:
        ah,aw=a.shape
    else:
        raise ValueError('a must be 2D or 3D.')
    if winsize[0]>ah or winsize[1]>aw:
        raise ValueError('winsize must be limited in %s.'%str((ah,aw)))
    
    rows=range(ah-winsize[0]+1)[::steps[0]]
    cols=range(aw-winsize[1]+1)[::steps[1]]
    x,y=np.meshgrid(rows,cols)
    t=np.stack((x,y))
    if flag==1:
        t=np.swapaxes(t,0,2)
    elif flag==0:
        t=np.transpose(t,(1,2,0))
    t=t.reshape(-1,2)
    t=np.ravel_multi_index(t.T,(ah,aw))
    q=(np.arange(winsize[1])[None,:]+np.array([aw*i for i in range(winsize[0])])[:,None]).flatten()
    index=t[:,None]+q[None,:]
    
    if len(a.shape)==3:
        wins=a.reshape(-1)[index+np.array([i*a.shape[1]*a.shape[2] for i in range(a.shape[0])])[:,None,None]]
    elif len(a.shape)==2:
        wins=a.flatten()[index]
    return wins,(rows,cols)

def pooling(a,winsize=(2,2),steps=None,method='max'):
    '''
    parameters:
        a、winsize、steps: see function [slide_wins].
        method: 'max' or 'average'. Default: 'max'.
    returns:
        return a 2D or 3D array. You can try to apply it to pool an image. A demo like this:
            >>> import matplotlib.pyplot as plt
            >>> import matplotlib.image as mimage
            >>> img=mimage.imread('lena.jpg')
            >>> img=np.rollaxis(img,2)
            >>> plt.imshow(np.rollaxis(pooling(img),0,3))
            >>> plt.show()
        Note that this function depends on [slide_wins], Unless there are no overlaps or gaps 
        between your pooling window.
    '''
    a=a.copy()
    dim=a.ndim
    if dim not in [2,3]:
        raise ValueError('a must be 2D or 3D.')
    
    if steps==None or steps==winsize:
        if winsize[0]>a.shape[-2] or winsize[1]>a.shape[-1]:
            raise ValueError('winsize must be limited in %s.'%str((a.shape[-2],a.shape[-1])))
        if dim==2:
            h,w=a.shape
            ysh=h%winsize[0]
            if ysh!=0:
                h-=ysh
                a=a[:-ysh]
            ysw=w%winsize[1]
            if ysw!=0:
                w-=ysw
                a=a[:,:-ysw]
            a=a.reshape(int(h/winsize[0]),winsize[0],int(w/winsize[1]),winsize[1])
            a=np.swapaxes(a,1,2)
            a=a.reshape(*a.shape[:-2],a.shape[-1]*a.shape[-2])
        elif dim==3:
            h,w=a.shape[1:]
            ysh=h%winsize[0]
            if ysh!=0:
                h-=ysh
                a=a[:,:-ysh]
            ysw=w%winsize[1]
            if ysw!=0:
                w-=ysw
                a=a[...,:-ysw]
            a=a.reshape(a.shape[0],int(h/winsize[0]),winsize[0],int(w/winsize[1]),winsize[1])
            a=np.swapaxes(a,2,3)
            a=a.reshape(*a.shape[:-2],a.shape[-1]*a.shape[-2])
        
        if method=='max':
            return np.max(a,axis=dim)
        elif method=='average':
            return np.average(a,axis=dim)
    else:
        r=slide_wins(a,winsize,steps)
        if method=='max':
            return np.max(r[0],axis=dim-1).reshape((-1,len(r[1][0]),len(r[1][1]))[3-dim:])
        elif method=='average':
            return np.average(r[0],axis=dim-1).reshape((-1,len(r[1][0]),len(r[1][1]))[3-dim:])

def get_hsv(imgs,colorBins=[8,8,8],winsize=(10,10),strides=(5,5),numScales=3):
    '''calculate LOMO's color histogram features(n,d1)'''
    hsv=np.zeros(imgs.shape)
    for i in range(imgs.shape[-1]):
        im=imgs[...,i]
        #im=retinex(im,[5,20])
        t=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        h,s,v=t[...,0]/255,t[...,1]/255,t[...,2]/255
        
        h=np.floor(h*colorBins[0])
        h[h>(colorBins[0]-1)]=colorBins[0]-1
        s=np.floor(s*colorBins[1])
        s[s>(colorBins[1]-1)]=colorBins[1]-1
        v=np.floor(v*colorBins[2])
        v[v>(colorBins[2]-1)]=colorBins[2]-1
        hsv[...,i]=np.concatenate((h[...,None],s[...,None],v[...,None]),2)
    hsv=hsv[:,:,2,:]*colorBins[1]*colorBins[0]+hsv[:,:,1,:]*colorBins[0]+hsv[:,:,0,:]
    
    hsv=np.rollaxis(hsv,2)
    features=None
    totalBins=colorBins[0]*colorBins[1]*colorBins[2]
    for i in range(numScales):
        wins,_=slide_wins(hsv,winsize,strides)
        wins=wins.astype('uint8')
        ko=np.zeros((wins.shape[0]*wins.shape[1],totalBins))
        k=0
        for j in range(wins.shape[0]):
            for l in range(wins.shape[1]):
                ri=cv2.calcHist([wins[j][l]],[0],None,[totalBins],[0,totalBins]).T
                ko[k]=ri
                k+=1
        wins=ko.reshape(*wins.shape[:2],-1)
        
        wins=wins.reshape(wins.shape[0],len(_[0]),len(_[1]),-1)
        fea=np.max(wins,axis=2).reshape(wins.shape[0],-1)
        if i==0:
            features=fea
        else:
            features=np.hstack((features,fea))
        hsv=pooling(hsv,method='average')
    print('calc HSV features(LOMO): ',features.shape)
    
    features=np.log(features+1)
    features=normc(features,1)
    return features

def get_siltp(imgs,winsize=(10,10),strides=(5,5),numScales=3,**kwargs):
    '''calculate LOMO's SILTP texture features(n,d2), note that kwargs will be passed to func calc_siltp(R,N,tau)'''
    imgHeight,imgWidth,_,numImgs=imgs.shape
    images=np.zeros((imgHeight,imgWidth,numImgs))
    for i in range(numImgs):
        I=cv2.cvtColor(imgs[...,i],cv2.COLOR_RGB2GRAY)
        images[:,:,i]=I.astype(np.float)
    features=None
    if kwargs.get('N')==None:
        kwargs['N']=4
    totalBins=3**kwargs['N']
    for i in range(numScales):
        siltp=calc_siltp2(images,**kwargs)
        siltp=np.rollaxis(siltp,2)
        wins,_=slide_wins(siltp,winsize,strides)
        
        wins=wins.astype('uint8')
        ko=np.zeros((wins.shape[0]*wins.shape[1],totalBins))
        k=0
        for j in range(wins.shape[0]):
            for l in range(wins.shape[1]):
                ri=cv2.calcHist([wins[j][l]],[0],None,[totalBins],[0,totalBins]).T
                ko[k]=ri
                k+=1
        wins=ko.reshape(*wins.shape[:2],-1)
        
        wins=wins.reshape(wins.shape[0],len(_[0]),len(_[1]),-1)
        fea=np.max(wins,axis=2).reshape(wins.shape[0],-1)
        if i==0:
            features=fea
        else:
            features=np.hstack((features,fea))
        images=np.rollaxis(images,2)
        images=pooling(images,method='average')
        images=np.rollaxis(images,0,3)
    print('calc SILTP features(LOMO): ',features.shape)
    
    features=np.log(features+1)
    features=normc(features,1)
    return features

if __name__=='__main__':
    pass