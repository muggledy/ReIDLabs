import numpy as np
import cv2
import matplotlib.pyplot as plt

def window_nd(a, window, steps = None, axis = None, outlist = False):
    """
    Create a windowed view over `n`-dimensional input that uses an 
    `m`-dimensional window, with `m <= n`

    Parameters
    ----------
    a : Array-like
        The array to create the view on

    window : tuple or int
        If int, the size of the window in `axis`, or in all dimensions if 
        `axis == None`

        If tuple, the shape of the desired window.  `window.size` must be:
            equal to `len(axis)` if `axis != None`, else 
            equal to `len(a.shape)`, or 1

    steps : tuple, int or None
        The offset between consecutive windows in desired dimension
        If None, offset is one in all dimensions
        If int, the offset for all windows over `axis`
        If tuple, the steps along each `axis`.  
            `len(steps)` must me equal to `len(axis)`

    axis : tuple, int or None
        The axes over which to apply the window
        If None, apply over all dimensions
        if tuple or int, the dimensions over which to apply the window

    outlist : boolean
        If output should be as list of windows.
        If False, it will be an array with 
            `a.nidim + 1 <= a_view.ndim <= a.ndim *2`.
        If True, output is a list of arrays with `a_view[0].ndim = a.ndim`
            Warning: this is a memory-intensive copy and not a view

    Returns
    -------

    a_view : ndarray
        A windowed view on the input array `a`, or copied list of windows

    Notes(すごい)
    -----
    Comes from: https://code.i-harness.com/zh-CN/q/2bd4c00
    """
    ashp = np.array(a.shape)

    if axis != None:
        axs = np.array(axis, ndmin = 1)
        assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
    else:
        axs = np.arange(ashp.size)

    window = np.array(window, ndmin = 1)
    assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
    wshp = ashp.copy()
    wshp[axs] = window
    assert np.all(wshp <= ashp), "Window is bigger than input array in axes"

    stp = np.ones_like(ashp)
    if steps:
        steps = np.array(steps, ndmin = 1)
        assert np.all(steps > 0), "Only positive steps allowed"
        assert (steps.size == axs.size) | (steps.size == 1), "Steps and axes don't match"
        stp[axs] = steps

    astr = np.array(a.strides)

    shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
    strides = tuple(astr * stp) + tuple(astr)

    as_strided = np.lib.stride_tricks.as_strided
    a_view = np.squeeze(as_strided(a, 
                                 shape = shape, 
                                 strides = strides))
    if outlist:
        return list(a_view.reshape((-1,) + tuple(wshp)))
    else:
        return a_view

def to_ycbcr(img):
    '''convert img from BGR to YCbCr'''
    return cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)[...,[0,2,1]]

def skin_detect(img):
    '''YCbCr Skin Detection Model(just for fun). refer to 
       https://blog.csdn.net/shadow_guo/article/details/43605117'''
    if isinstance(img,str): #if an image path
        img=cv2.imread(img)
    h,w=img.shape[:2]
    gamma=0.95
    img=(img**gamma).astype(np.int).astype(np.uint8) #light compensation
    imgYcc=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    imgSkin=img.copy()
    Wcb=46.97;Wcr=38.76;WHCb=14;WHCr=10;WLCb=23;WLCr=20
    Ymin=16;Ymax=235;Kl=125;Kh=188
    WCb=np.zeros((h,w))
    WCr=np.zeros((h,w))
    CbCenter=np.zeros((h,w))
    CrCenter=np.zeros((h,w))
    skin=np.zeros((h,w),np.int)
    imgYcc=imgYcc.astype('double') #shed my blood and tears, numeric 
                                   #overflow if with type uint8
    Y,Cr,Cb=imgYcc[...,0],imgYcc[...,1],imgYcc[...,2]
    t=Y<Kl
    WCr[t]=WLCr+(Y[t]-Ymin)*(Wcr-WLCr)/(Kl-Ymin)
    WCb[t]=WLCb+(Y[t]-Ymin)*(Wcb-WLCb)/(Kl-Ymin)
    CrCenter[t]=154-(Kl-Y[t])*(154-144)/(Kl-Ymin)
    CbCenter[t]=108+(Kl-Y[t])*(118-108)/(Kl-Ymin)
    t=Y>Kh
    WCr[t]=WHCr+(Y[t]-Ymax)*(Wcr-WHCr)/(Ymax-Kh)
    WCb[t]=WHCb+(Y[t]-Ymax)*(Wcb-WHCb)/(Ymax-Kh)
    CrCenter[t]=154+(Y[t]-Kh)*(154-132)/(Ymax-Kh)
    CbCenter[t]=108+(Y[t]-Kh)*(118-108)/(Ymax-Kh)
    t=(Y<Kl)|(Y>Kh)
    Cr[t]=(Cr[t]-CrCenter[t])*Wcr/WCr[t]+154
    Cb[t]=(Cb[t]-CbCenter[t])*Wcb/WCb[t]+108
    t=(Cb>77)&(Cb<127)&(Cr>133)&(Cr<173)
    skin[t]=1
    imgSkin[skin==0,:]=0
    return imgSkin

def get_patches(X, window, steps):
    '''create slide windows of X(h,w,channel), note that if window or steps 
       is an integer x, it will be extended to (x,x)'''
    if len(X.shape) != 3:
        raise ValueError('X\'s shape should be (h,w,c)!')
    patches = window_nd(X, window, steps, axis=(0,1))
    if window == 1 or window == (1,1):
        s = patches.shape
        s = tuple([*s[:-1], 1, 1, s[-1]])
        patches = patches.reshape(s)
    return patches

def plot_patches(patches):
    '''for the result of func get_patches, if the arg X of get_patches has 
       only 3 channels, i.e. an image, we can plot it with matplotlib'''
    if len(patches.shape)!=5 or patches.shape[-1]!=3:
        raise ValueError('invalid patches value!')
    patches=patches.copy()
    patches[...,[0,-1],:,:]=255 #white split lines
    patches[...,[0,-1],:]=255
    
    patch_h,patch_w=patches[0][0].shape[:2]
    rows,cols=patches.shape[:2]
    
    dst_img = np.zeros((patch_h * rows, patch_w * cols, 3), dtype = 'uint8')
    patches=np.rollaxis(patches,0,5)
    patches=np.rollaxis(patches,0,5)
    for col in range(cols): #i can't do better
        col_imgs = patches[..., col]
        col_imgs = np.rollaxis(col_imgs, -1)
        col_imgs = col_imgs.reshape(-1, patch_w, 3)
        dst_img[:, col * patch_w:(col + 1) * patch_w, :] = col_imgs
    
    plt.imshow(dst_img[...,::-1])
    plt.show()

if __name__=='__main__':
    import os.path
    cwd=os.path.dirname(__file__)
    '''
    img=cv2.imread(os.path.join(cwd,'test.webp')) #attrib
    plt.subplot(121)
    plt.imshow(img[...,::-1])
    plt.subplot(122)
    plt.imshow(skin_detect(img)[...,::-1])
    plt.show()
    '''
    img=cv2.imread(os.path.join(cwd,'../../../images/VIPeR.v1.0/cam_a/000_45.bmp'))
    #patches=get_patches(img,20,5) #a demo for patches
    patches=get_patches(img,(img.shape[0]/4,img.shape[1]),(img.shape[0]/8,img.shape[1]))[:,None,:,:,:] #a demo for horizontal strips
    plot_patches(patches)
    