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

#随机擦除可以提升模型的抗遮挡能力，且与随机裁剪、随机水平翻转具有一定的互补性
#由于reid数据及来自不同场景下的摄像头，因光照变化以及摄像头参数设置等原因，数据集中的图片难免存在
#着由于场景变化过大而难以识别的图片，针对这类情况，可以对数据集做色彩抖动处理，在扩充数据集的同时，
#让模型适应复杂场景带来的变化。色彩抖动指的是随机改变图片的亮度、对比度和饱和度，色彩抖动通过调整这
#三个不同的指标，模拟不同的光照变化，扩充更多背景亮度差异比较明显的图片

class ColorJitter:
    '''https://github.com/michuanhaohao/keras_reid/blob/7bc111887fb8c82b68ed301c75c3e06a0e39bc1a/aug.py
       测试似乎并未带来提升，反而降低了1个点左右？
    '''

    def __init__(self, probability=0.5, brightness=0.4, contrast=0.4, saturation=0.4):
        self.probability=probability
        self.brightness=brightness
        self.contrast=contrast
        self.saturation=saturation

    def __call__(self,img):
        if random.uniform(0, 1) > self.probability:
            return img
        img=np.array(img)
        rng = np.random.RandomState()
        img=self._augment(img, rng, self.brightness, self.contrast, self.saturation)
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def _augment(img, rng, brightness, contrast, saturation):
        def grayscale(img):
            w = np.array([0.114, 0.587, 0.299]).reshape(1, 1, 3)
            gs = np.zeros(img.shape[:2])
            gs = (img * w).sum(axis=2, keepdims=True)
            return gs
        
        def brightness_aug(img, val):
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha
            return img
        def contrast_aug(img, val):
            gs = grayscale(img)
            gs[:] = gs.mean()
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)
            return img
        def saturation_aug(img, val):
            gs = grayscale(img)
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)
            return img
        
        def color_jitter(img, brightness, contrast, saturation):
            augs = [(brightness_aug, brightness),
                    (contrast_aug, contrast),
                    (saturation_aug, saturation)]
            rng.shuffle(augs)
            for aug, val in augs:
                img = aug(img, val)
            return img
        
        def lighting(img, std):
            eigval = np.array([0.2175, 0.0188, 0.0045])
            eigvec = np.array([
                [-0.5836, -0.6948,  0.4203],
                [-0.5808, -0.0045, -0.8140],
                [-0.5675, 0.7192, 0.4009],
            ])
            if std == 0:
                return img
            alpha = rng.randn(3) * std
            bgr = eigvec * alpha.reshape(1, 3) * eigval.reshape(1, 3)
            bgr = bgr.sum(axis=1).reshape(1, 1, 3)
            img = img + bgr
            return img

        img = color_jitter(img, brightness=brightness, contrast=contrast, saturation=saturation)
        # img = lighting(img, 0.1)
        img = np.minimum(255.0, np.maximum(0, img))
        return img#.astype('float32') #return h*w*3

class Lighting:
    '''Lighting with Retinex（完全失败，降低了5、6个点）'''
    def __init__(self,probability=0.5,**kwargs):
        self.probability=probability
        self.kwargs=kwargs

    def __call__(self,img):
        if random.uniform(0, 1) > self.probability:
            return img
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
    img=ColorJitter()(img)
    import torchvision.transforms as T
    img=T.RandomHorizontalFlip()(img)
    img=RandomErasing()(img)
    plt.subplot(122)
    plt.imshow(img)
    plt.show()
    #一个很莫名其妙的警告：libpng warning: iCCP: cHRM chunk does not match sRGB
    #竟然是由QQ输入法引起的？关闭就没有警告了，what fuck