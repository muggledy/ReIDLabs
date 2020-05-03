'''
函数 LC,HC,FT,RC 代码来自：https://blog.csdn.net/wsp_1138886114/article/details/102560328
作者：SongpingWang

图像显著性检测论文：https://blog.csdn.net/u010736662/article/details/88930849
'''

import matplotlib.pyplot as plt
import os.path
import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
from .rc_region.rc_region import segmentImage, Build_Regions_Contrast, Quantize, Smooth # 基于图的图像分割, 创建区域并形成区域对比度, 颜色量化, 区域平滑

def diag_sym_matrix(k=256):
    base_matrix = np.zeros((k,k))
    base_line = np.array(range(k))
    base_matrix[0] = base_line
    for i in range(1,k):
        base_matrix[i] = np.roll(base_line,i)
    base_matrix_triu = np.triu(base_matrix)
    return base_matrix_triu + base_matrix_triu.T

def cal_dist(hist):
    Diag_sym = diag_sym_matrix(k=256)
    hist_reshape = hist.reshape(1,-1)
    hist_reshape = np.tile(hist_reshape, (256, 1))
    return np.sum(Diag_sym*hist_reshape,axis=1)

def LC(image_gray):
    '''基于全局对比度图像显著性检测（LC）'''
    image_height,image_width = image_gray.shape[:2]
    hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])
    gray_dist = cal_dist(hist_array)

    image_gray_value = image_gray.reshape(1,-1)[0]
    image_gray_copy = [(lambda x: gray_dist[x]) (x)  for x in image_gray_value]
    image_gray_copy = np.array(image_gray_copy).reshape(image_height,image_width)
    image_gray_copy = (image_gray_copy-np.min(image_gray_copy))/(np.max(image_gray_copy)-np.min(image_gray_copy))
    return image_gray_copy

def HC(img,delta=0.25):
    '''基于直方图对比度的显著性检测（HC）'''
    img_float = img.astype(np.float32)
    img_float = img_float / 255.0
    binN, idx1i, binColor3f, colorNums1i = Quantize(img_float) # 颜色量化
    binColor3f = cv2.cvtColor(binColor3f, cv2.COLOR_BGR2Lab) # 颜色空间：BGR2Lab
    weight1f = np.zeros(colorNums1i.shape, np.float32)
    cv2.normalize(colorNums1i.astype(np.float32), weight1f, 1, 0, cv2.NORM_L1) # 相邻色彩相关权重

    binColor3f_reshape = binColor3f.reshape(-1, 3)[:binN]
    similar_dist = squareform(pdist(binColor3f_reshape))
    similar_dist_sort = np.sort(similar_dist)
    similar_dist_argsort = np.argsort(similar_dist)

    weight1f = np.tile(weight1f, (binN, 1))
    color_weight_dist = np.sum(np.multiply(weight1f, similar_dist), axis=1) # 颜色距离的权重分配

    colorSal = np.zeros((1, binN), np.float64)
    if colorSal.shape[1] < 2:
        return
    tmpNum = int(np.round(binN * delta)) # tmpNum 占比0.25的变化的颜色值数量
    n = tmpNum if tmpNum > 2 else 2

    similar_nVal = similar_dist_sort[:, :n]
    totalDist_similar = np.sum(similar_nVal, axis=1)
    every_Dist = np.tile(totalDist_similar[:, np.newaxis], (1, n)) - similar_nVal

    idx = similar_dist_argsort[:, :n]
    val_n = np.take(color_weight_dist,idx) # 获取占比前0.25的颜色权重距离

    valCrnt = np.sum(val_n[:, :n] * every_Dist, axis=1)
    newSal_img = valCrnt / (totalDist_similar * n)
    cv2.normalize(newSal_img, newSal_img, 0, 1, cv2.NORM_MINMAX) # 归一化
    salHC_img = np.take(newSal_img,idx1i)
    cv2.GaussianBlur(salHC_img, (3, 3), 0, salHC_img)
    cv2.normalize(salHC_img, salHC_img, 0, 1, cv2.NORM_MINMAX)
    return salHC_img

def FT(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img,(5,5), 0)
    gray_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l_mean = np.mean(gray_lab[:,:,0])
    a_mean = np.mean(gray_lab[:,:,1])
    b_mean = np.mean(gray_lab[:,:,2])
    lab = np.square(gray_lab- np.array([l_mean, a_mean, b_mean]))
    lab = np.sum(lab,axis=2)
    lab = lab/np.max(lab)
    return lab

def SR(mat):
    '''Spectral Residual（谱残差），来自：https://github.com/MirusUmbra/visual-saliency-detection
       参考：https://blog.csdn.net/weixin_42647783/article/details/81415480'''
    fourier = cv2.dft(np.float32(mat), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 实部和虚部平方和再开方，得到合并的频域用作后续运算
    # 虽然论文提到全程计算只使用振幅谱,相位谱不变,但是实际计算振幅谱的对数时出现超过定义域因而无法计算对数, 并且论文
    # 作者实际也是计算振幅+相位, 所以暂时与作者代码保持一致
    # Different with paper, the author actually used both imaginary and real part of the frequency domain.
    re, im = cv2.split(fourier)
    base = (re ** 2 + im ** 2) ** 0.5
    # 对数谱
    log = cv2.log(base)
    # 平滑曲线
    blur = cv2.blur(log, (7, 7))
    # 显著性余谱
    # Get residual
    residual = log - blur
    # 指数还原对数谱
    # Restore
    residual = cv2.exp(residual)
    # 求原频域上实虚的夹角, 利用夹角还原实虚
    sin = im / base
    cos = re / base 
    re = residual * cos
    im = residual * sin
    # 傅里叶逆变换
    fourier = cv2.merge((re, im))
    inverse = cv2.dft(fourier, flags=cv2.DFT_INVERSE + cv2.DFT_REAL_OUTPUT)
    # 优化结果显示
    min_v, max_v, _, _ = cv2.minMaxLoc(inverse)
    _, thre = cv2.threshold(inverse, 0, 255, cv2.THRESH_TOZERO)
    thre = thre * (255 / max_v)
    res = cv2.GaussianBlur(thre, (7, 7), 0)
    return res

def RC(img, sigmaDist=0.4, segK=20, segMinSize=200, segSigma=0.5):
    img3f=img.astype(np.float32)/255
    
    height, width = img3f.shape[:2]
    imgLab3f = img3f.copy()
    cv2.cvtColor(img3f, cv2.COLOR_BGR2Lab, imgLab3f)
    smImg3f = np.zeros(imgLab3f.shape, dtype=imgLab3f.dtype)
    cv2.GaussianBlur(imgLab3f, (0, 0), 0.5, dst=smImg3f, borderType=cv2.BORDER_REPLICATE)

    regNum, regIdx1i = segmentImage(smImg3f, seg_K=20.0, min_size=200)
    Quatizenum, colorIdx1i, color3fv, tmp = Quantize(img3f)

    if Quatizenum == 2:
        sal = colorIdx1i.copy()
        cv2.compare(colorIdx1i, 1, cv2.CMP_EQ, sal)
        sal = sal.astype(np.float32)
        mn = np.min(sal)
        mx = np.max(sal)
        sal = (sal - mn) * 255 / (mx - mn)
        return sal
    if Quatizenum <= 2:
        return np.zeros(img3f.shape, img3f.dtype)
    cv2.cvtColor(color3fv, cv2.COLOR_BGR2Lab, color3fv)
    regSal1v,bdgReg1u = Build_Regions_Contrast(regNum,regIdx1i,colorIdx1i,color3fv,sigmaDist,0.02, 0.4)
    sal1f = np.zeros((height, width), img3f.dtype)
    cv2.normalize(regSal1v, regSal1v, 0, 1, cv2.NORM_MINMAX)
    sal1f = np.take(regSal1v[0],regIdx1i)
    idxs = np.where(bdgReg1u == 255)
    sal1f[idxs] = 0
    sal1f = Smooth(img3f, sal1f, regIdx1i, regNum,delta=0.1)
    sal1f[idxs] = 0
    cv2.GaussianBlur(sal1f, (3, 3), 0, sal1f)
    return sal1f

def linear_stretch_255(img):
    mmin=np.min(img,axis=(0,1),keepdims=True)
    mmax=np.max(img,axis=(0,1),keepdims=True)
    img=(img-mmin)/(mmax-mmin)*255
    return img.astype('uint8')

def threshold_SR(sal):
    mask=np.zeros(sal.shape)
    mask[sal>(np.mean(sal)*3)]=255
    return linear_stretch_255(mask)

def threshold_RC(sal):
    mask=np.zeros(sal.shape)
    mask[sal>(sal.max()+sal.min())*0.2]=255 #/1.8
    return linear_stretch_255(mask)


def seg_img_with_mask(img,mask,plot=True):
    ret=cv2.bitwise_and(img,img,mask=mask)
    if plot:
        plt.subplot(121)
        plt.imshow(mask,cmap='gray')
        plt.subplot(122)
        plt.imshow(ret[...,::-1])
        plt.show()
    else:
        return ret

if __name__=='__main__':
    img = cv2.imread(os.path.join(os.path.dirname(__file__),'./imgs/1.png'))
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    plt.subplot(2,3,1)
    plt.imshow(img[...,::-1])

    lc_img=LC(img_gray)
    lc_img=linear_stretch_255(lc_img)
    plt.subplot(2,3,2)
    plt.imshow(lc_img,cmap='gray')

    hc_img = HC(img)
    hc_img=linear_stretch_255(hc_img)
    plt.subplot(2,3,3)
    plt.imshow(hc_img,cmap='gray')

    ft_img=FT(img)
    ft_img=linear_stretch_255(ft_img)
    plt.subplot(2,3,4)
    plt.imshow(ft_img,cmap='gray')

    sr_img=SR(img_gray)
    sr_img=linear_stretch_255(sr_img)
    plt.subplot(2,3,5)
    plt.imshow(sr_img,cmap='gray')
    
    rc_img=RC(img)
    rc_img=linear_stretch_255(rc_img)
    plt.subplot(2,3,6)
    plt.imshow(rc_img,cmap='gray')
    plt.show()
    
    '''
    sr_img=SR(img_gray)
    seg_img_with_mask(img,threshold_SR(sr_img))
    '''
    '''
    rc_img=RC(img)
    seg_img_with_mask(img,threshold_RC(rc_img))
    '''
    '''
    hc_img=HC(img)
    seg_img_with_mask(img,threshold_SR(hc_img))
    '''