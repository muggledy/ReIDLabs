import random
import math
import numpy as np
from PIL import Image
import cv2
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../'))
from zoo.tools import gauss_blur

class RandomErasing(object): #comes from 
    #https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/transforms.py
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, \
                       mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        img=np.array(img)
        if len(img.shape)==2: #如果是单通道图
            img=img[...,None]
        img=np.rollaxis(img,2)

        for attempt in range(100):
            area = img.shape[1] * img.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return Image.fromarray(np.rollaxis(img,0,3))
        return Image.fromarray(np.rollaxis(img,0,3))

class Lighting:
    '''Lighting with Retinex'''
    def __init__(self,**kwargs):
        self.kwargs=kwargs

    def __call__(self,img):
        img=np.array(img)
        return Image.fromarray(self.retinex_gimp(img,**self.kwargs))
    
    @staticmethod
    def retinex_gimp(img,sigmas=[12,80,250],dynamic=2):
        def MultiScaleRetinex(img,sigmas=[15,80,250],weights=None,flag=True):
            if weights==None:
                weights=np.ones(len(sigmas))/len(sigmas)
            elif not abs(sum(weights)-1)<0.00001:
                raise ValueError('sum of weights must be 1!')
            r=np.zeros(img.shape,dtype='double')
            img=img.astype('double')
            for i,sigma in enumerate(sigmas):
                r+=(np.log(img+1)-np.log(gauss_blur(img,sigma)+1))*weights[i]
            if flag:
                mmin=np.min(r,axis=(0,1),keepdims=True)
                mmax=np.max(r,axis=(0,1),keepdims=True)
                r=(r-mmin)/(mmax-mmin)*255 #maybe indispensable when used in MSRCR or Gimp, make pic vibrant
                r=r.astype('uint8')
            return r
        alpha,gain,offset=128,1,0
        img=img.astype('double')+1
        csum_log=np.log(np.sum(img,axis=2))
        msr=MultiScaleRetinex(img-1,sigmas)
        r=gain*(np.log(alpha*img)-csum_log[...,None])*msr+offset
        mean=np.mean(r,axis=(0,1),keepdims=True)
        var=np.sqrt(np.sum((r-mean)**2,axis=(0,1),keepdims=True)/r[...,0].size)
        mmin=mean-dynamic*var
        mmax=mean+dynamic*var
        stretch=(r-mmin)/(mmax-mmin)*255
        stretch[stretch>255]=255
        stretch[stretch<0]=0
        return stretch.astype('uint8')

if __name__=='__main__':
    import os.path
    img_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../images/VIPeR.v1.0/cam_a/448_0.bmp')
    from PIL import Image
    img=Image.open(img_path).convert('RGB')
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(img)
    # img=Lighting()(img)
    import torchvision.transforms as T
    img=T.RandomHorizontalFlip()(img)
    img=RandomErasing()(img)
    # img=Lighting()(img)
    plt.subplot(122)
    plt.imshow(img)
    plt.show()