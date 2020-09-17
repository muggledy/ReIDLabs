import torch as pt
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../'))
from gog.utils import plot_patches
from zoo.cprint import colors as COLORS
from deep.data_manager import load_query_gallery_dataset,Ana78, \
    get_query_gallery_dataset
from deep.data_loader import load_dataset
from deep.eval_metric import eval_cmc_map
from deep.test import test
import matplotlib.pyplot as plt
import cv2
from pprint import pprint

def set_img_color_frame(img,colors,widths=None):
    '''例如colors=['black','red'],widths=[1,2]表示先绘制最外层宽度为
       1的黑色边框，再绘制靠里一层的宽度为2的红色边框'''
    if not isinstance(colors,(list,tuple)):
        colors=[colors]
    if widths is None:
        widths=[1]*len(colors)
    elif not isinstance(widths,(list,tuple)):
        widths=[widths]
    if len(colors)!=len(widths):
        raise ValueError('The num of colors must match widths!')
    inds=np.cumsum([0]+widths)
    h,w,c=img.shape
    def _f(_img,_color,_width=2):
        '''img's shape is (h,w,3)，color可以是常见的颜色名，如'red'，
        也可以是RGB值，如[0,255,255]。注意这边img是用cv2读取的，是
        BGR通道，而color是RGB通道，所以赋值时要注意换一下顺序'''
        if isinstance(_color,str): #通过colors.keys()查看所有支持的颜色名
            _color=COLORS[_color][2]
        _color=_color[::-1] #RGB->BGR
        _img[0:_width,:,:]=_color
        _img[-_width:,:,:]=_color
        _img[:,0:_width,:]=_color
        _img[:,-_width:,:]=_color
    for i in range(len(colors)):
        _f(img[inds[i]:h-inds[i],inds[i]:w-inds[i],:],colors[i],widths[i])

def plot_match(net,query_dir=None,gallery_dir=None,checkpoint=None,galfeas_path=None,query_num=10, \
               gallery_num=20,batch_size=(1,32),resize=(128,256),re_rank=False,**kwargs):
    '''注意query最多query_num个，如果query_dir下超过query_num个图片，则随机挑选query_num个，另外只显示
       前gallery_num个gallery匹配。resize是因为图像的尺寸可能不一致'''
    querySet=kwargs.get('querySet')
    gallerySet=kwargs.get('gallerySet')
    query_gallery=kwargs.get('query_gallery')
    analyse=kwargs.get('analyse',Ana78(r'([-\d]+)_c(\d)')) #默认值是为market1501保留
    if query_dir and gallery_dir: #在某些特殊的情况下，query_gallery难以自动构造，譬如数据集图片名称中只含行人ID不含
                          #摄像头ID，相应的analyse也只能获取行人ID，摄像头ID必须手动设置，此时我们直接在外面
                          #构造好query_gallery，主要包括两个属性，一是querySet，一是gallerySet，然后传递给plot_match
                          #，示例见deep_jstl.py
        query_gallery=load_query_gallery_dataset(query_dir,gallery_dir,query_num,analyse)
    elif query_gallery:
        if query_num<len(query_gallery.querySet):
            selectedQuerySet=[]
            for i in np.random.permutation(len(query_gallery.querySet))[:query_num]:
                    selectedQuerySet.append(query_gallery.querySet[i])
            query_gallery.querySet=selectedQuerySet
    elif querySet and gallerySet:
        query_gallery=get_query_gallery_dataset(querySet,gallerySet,query_num)
    else:
        raise ValueError('Must give query_dir/gallery_dir or querySet/gallerySet or query_gallery!')
    # query_gallery.print_info()
    query_iter,gallery_iter=load_dataset(query_gallery,test_batch_size=batch_size,notrain=True)

    if checkpoint is not None and checkpoint.loaded:
        net.load_state_dict(checkpoint.states_info_etc['state']) #注意必须在执行net=DataParallel(net)之后加载参数

    re_calc_gal_fea=kwargs.get('re_calc_gal_fea',False)
    match_inds=test(net,query_iter,galfeas_path if not re_calc_gal_fea and galfeas_path is not None and \
        os.path.exists(galfeas_path) \
        else gallery_iter,eval_cmc_map,calc_dist_funcs=kwargs.get('calc_dist_funcs'), \
        alphas=kwargs.get('alphas'),save_galFea=galfeas_path,re_rank=re_rank)[:,:gallery_num]
    
    imgs=[]
    first_pairs=[]
    for i,(query_img_path,query_pid,query_cid) in enumerate(query_gallery.querySet):
        imgs.append(cv2.resize(cv2.imread(query_img_path),resize))
        for j in range(gallery_num):
            gallery_img_path,gallery_pid,gallery_cid=query_gallery.gallerySet[match_inds[i][j]]
            gallery_img=cv2.resize(cv2.imread(gallery_img_path),resize)
            if query_pid==gallery_pid: #同一ID即匹配，标记为蓝色边框
                set_img_color_frame(gallery_img,'Blue',2)
                if query_cid!=gallery_cid: #若进一步不是同一摄像头，则置为红色，这也是我们最关注的
                    set_img_color_frame(gallery_img,'Red',2)
            else: #ID不匹配，置为绿色
                set_img_color_frame(gallery_img,'Green',2)
            imgs.append(gallery_img)
            if j==0:
                first_pairs.append((os.path.normpath(query_img_path),os.path.normpath(gallery_img_path)))
    pprint(first_pairs) #列表中每个元素都是一个二元组，第一个值是query，第二个值是rank-1的gallery
    imgs=np.array(imgs)
    imgs=imgs.reshape(query_num,gallery_num+1,*(imgs.shape[1:]))
    plot_patches(imgs,line=False,axes=False,x_label='gallery(rank1-%d)'%gallery_num,y_label='query')

if __name__=='__main__':
    from models.utils import CheckPoint
    from models.ResNet import ResNet50_Aligned
    from deep.train import setup_seed
    setup_seed(0)
    pathbase=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../images/Market-1501-v15.09.15')
    query_dir=os.path.join(pathbase,'./query')
    gallery_dir=os.path.join(pathbase,'./bounding_box_test')
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../data/market1501_resnetAligned_gallery.mat')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_Aligned.tar')
    net=ResNet50_Aligned(751)
    plot_match(net,query_dir,gallery_dir,checkpoint,gal_savedir,re_calc_gal_fea=False,re_rank=False)