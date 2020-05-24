import sys
import os.path
cwd=os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(cwd,'../')))
from lomo.lomo import load_data
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic,mark_boundaries
from skimage import io
import os.path
import cv2

def mean(imgs):
    channel_nums=imgs.shape[-1]*imgs.shape[-2]
    imgs=np.rollaxis(imgs,-1)
    imgs=np.dstack(imgs)
    print(imgs.shape)
    imgs=(np.sum(imgs,axis=2)/channel_nums).astype('uint8')
    return imgs

def plot_mean(*imgs):
    n=len(imgs)
    for i in range(n):
        plt.subplot(1,n,i+1)
        img=imgs[i]
        #_, img = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY_INV)
        #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 121, 3)
        #_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        plt.imshow(img,cmap='gray')
    plt.show()

def show_super_pixel_seg(img):
    img=cv2.resize(img,(0,0),fx=3,fy=3)

    segments = slic(img, n_segments=60, compactness=16) #https://blog.csdn.net/weixin_41819299/article/details/84641465
    out=mark_boundaries(img,segments)

    plt.subplot(121)
    plt.title("n_segments=60")
    plt.imshow(out[...,::-1])

    segments2 = slic(img, n_segments=300, compactness=16)
    out2=mark_boundaries(img,segments2)
    plt.subplot(122)
    plt.title("n_segments=300")
    plt.imshow(out2[...,::-1])

    plt.show()

if __name__=='__main__':
    imgs_a=load_data(os.path.join(os.path.dirname(__file__),'../../../images/VIPeR.v1.0/cam_a'))
    imgs_b=load_data(os.path.join(os.path.dirname(__file__),'../../../images/VIPeR.v1.0/cam_b'))
    '''
    a=mean(imgs_a)
    b=mean(imgs_b)
    ab=mean(np.concatenate((imgs_a,imgs_b),axis=-1))
    plot_mean(a,b,ab)
    '''
    show_super_pixel_seg(imgs_a[...,0])
