'''
无监督模型迁移，在多个无关数据集上训练，在目标数据集上测试。目前来看效果很差，譬如在
cuhk01,cuhk03,market1501数据集上训练，在viper上测试，结果rank1只有22.31%，mAP为26.51
lr=0.0003,num_epochs=20,step_size=15(StepLR),gamma=0.1（实验1）
用时1.73小时
实验2：在cuhk01,cuhk03,market1501,prid2011,3dpes,ilids,shinpuhkan数据集上训练，
在viper上测试，rank1为29.43%，mAP为33.81，训练参数：lr=0.0003,num_epochs=50,采用
MultiStepLR([17,25,30])进行学习率衰减，用时5.79小时
基于实验2（将viper和market1501角色换一下），在market1501上的迁移结果为：
Rank-1:42.84% Rank-5:62.50% Rank-10:70.37% Rank-20:77.26% Rank-100:88.90% 
mAP:20.62
https://github.com/Cysu/dgd_person_reid
'''

from initial import *
from deep.models.ResNet import ResNet50_Classify
from deep.train import train,setup_seed
from deep.test import test
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint
from deep.data_loader import load_dataset,load_train_iter,load_query_or_gallery_iter
from deep.data_manager import MixDataSets,load_query_gallery_dataset,Ana716,Ana78, \
    process_cuhk01,process_cuhk03,process_prid2011,process_3dpes,process_ilids, \
    process_shinpuhkan
from deep.plot_match import plot_match
import torch as pt
import torch.nn as nn

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet56_jstl(no_viper).tar')

    mixdatasets=MixDataSets(
        ('cuhk01',os.path.join(dataset_dir,'CUHK01')), \
        ('cuhk03',os.path.join(dataset_dir,'cuhk03_images/detected')), \
        ('market1501',os.path.join(dataset_dir,'Market-1501-v15.09.15')), \
        ('prid2011',os.path.join(dataset_dir,'./prid2011/single_shot')), \
        ('3dpes',os.path.join(dataset_dir,'./3DPeS/RGB/')),\
        ('ilids',os.path.join(dataset_dir,'./i-LIDS/')), \
        ('shinpuhkan',os.path.join(dataset_dir,'Shinpuhkan/images/'))
    )

    batch_size=32
    train_iter=load_train_iter(mixdatasets.dataset)
    re_calc_gal_feat=True #之前可能保存了gallery特征，是否重新计算

    train_num_ids=len(set(list(zip(*mixdatasets.dataset))[1]))
    # print('混合训练集总计行人类别数: %d, 图片总数: %d'%(train_num_ids,len(mixdatasets.dataset)))
    mixdatasets.print_info()
    net=ResNet50_Classify(train_num_ids)
    loss=nn.CrossEntropyLoss()
    lr,num_epochs=0.0003,50
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    # scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.1)
    scheduler=pt.optim.lr_scheduler.MultiStepLR(optimizer,[17,25,30],gamma=0.1)
    train(net,train_iter,(loss,),optimizer,num_epochs,scheduler,checkpoint=checkpoint)

    print('Test on target dataset VIPeR:')
    query_gallery=load_query_gallery_dataset(os.path.join(dataset_dir,'./VIPeR.v1.0/cam_a'), \
        os.path.join(dataset_dir,'./VIPeR.v1.0/cam_b'),query_num=None,analyse=Ana716(r'(\d{,3})_'))
    query_imgs,query_pids,_=list(zip(*(query_gallery.querySet)))
    query_gallery.querySet=list(zip(query_imgs,query_pids,[0]*632)) #对VIPeR数据集必须手动设置摄像头ID
    gallery_imgs,gallery_pids,_=list(zip(*(query_gallery.gallerySet)))
    query_gallery.gallerySet=list(zip(gallery_imgs,gallery_pids,[1]*632))
    query_iter,gallery_iter=load_dataset(query_gallery,test_batch_size=batch_size,notrain=True)
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/viper_jstl_gallery.mat')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) and not re_calc_gal_feat \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,query_gallery=query_gallery,checkpoint=checkpoint,galfeas_path=gal_savedir)

    #在训练集market1501、cuhk01、03上测试，可想而知rank-1和mAP都会非常高
    print('Evaluate on training set Market1501:')
    query_dir=os.path.join(dataset_dir,'./Market-1501-v15.09.15/query')
    gallery_dir=os.path.join(dataset_dir,'./Market-1501-v15.09.15/bounding_box_test')
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/market1501_jstl_gallery.mat')
    query_gallery=load_query_gallery_dataset(query_dir,gallery_dir,query_num=None,analyse=Ana78(r'([-\d]+)_c(\d)'))
    query_iter,gallery_iter=load_dataset(query_gallery,test_batch_size=batch_size,notrain=True)
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) and not re_calc_gal_feat \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,query_gallery=query_gallery,checkpoint=checkpoint,galfeas_path=gal_savedir)

    print('Evaluate on training set CUHK01:')
    dataset=process_cuhk01(os.path.join(dataset_dir,'./CUHK01'))
    querySet,gallerySet=[],[]
    for img,pid,cid in dataset:
        if cid==0:
            querySet.append((img,pid,cid))
        elif cid==1:
            gallerySet.append((img,pid,cid))
    query_iter=load_query_or_gallery_iter(querySet)
    gallery_iter=load_query_or_gallery_iter(gallerySet)
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/cuhk01_jstl_gallery.mat')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) and not re_calc_gal_feat \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,querySet=querySet,gallerySet=gallerySet,checkpoint=checkpoint,galfeas_path=gal_savedir)

    print('Evaluate on training set CUHK03:')
    dataset=process_cuhk03(os.path.join(dataset_dir,'./cuhk03_images/detected'))
    querySet,gallerySet=[],[]
    for img,pid,cid in dataset:
        if cid in [0,2,4,6,8]:
            querySet.append((img,pid,cid))
        elif cid in [1,3,5,7,9]:
            gallerySet.append((img,pid,cid))
    query_iter=load_query_or_gallery_iter(querySet)
    gallery_iter=load_query_or_gallery_iter(gallerySet)
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/cuhk03_jstl_gallery.mat')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) and not re_calc_gal_feat \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,querySet=querySet,gallerySet=gallerySet,checkpoint=checkpoint,galfeas_path=gal_savedir)

    print('Evaluate on training set PRID2011:')
    dataset=process_prid2011(os.path.join(dataset_dir,'./prid2011/single_shot'))
    querySet,gallerySet=[],[]
    for img,pid,cid in dataset:
        if cid==0:
            querySet.append((img,pid,cid))
        if cid==1:
            gallerySet.append((img,pid,cid))
    query_iter=load_query_or_gallery_iter(querySet)
    gallery_iter=load_query_or_gallery_iter(gallerySet)
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/prid2011_jstl_gallery.mat')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) and not re_calc_gal_feat \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,querySet=querySet,gallerySet=gallerySet,checkpoint=checkpoint,galfeas_path=gal_savedir)

    print('Evaluate on training set 3DPeS:')
    dataset=process_3dpes(os.path.join(dataset_dir,'./3DPeS/RGB/'))
    querySet,gallerySet=[],[]
    for img,pid,cid in dataset:
        if cid==0:
            querySet.append((img,pid,cid))
        if cid==1:
            gallerySet.append((img,pid,cid))
    query_iter=load_query_or_gallery_iter(querySet)
    gallery_iter=load_query_or_gallery_iter(gallerySet)
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/3dpes_jstl_gallery.mat')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) and not re_calc_gal_feat \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,querySet=querySet,gallerySet=gallerySet,checkpoint=checkpoint,galfeas_path=gal_savedir)

    print('Evaluate on training set i-LIDS:')
    dataset=process_ilids(os.path.join(dataset_dir,'./i-LIDS/'))
    querySet,gallerySet=[],[]
    for img,pid,cid in dataset:
        if cid==0:
            querySet.append((img,pid,cid))
        if cid==1:
            gallerySet.append((img,pid,cid))
    query_iter=load_query_or_gallery_iter(querySet)
    gallery_iter=load_query_or_gallery_iter(gallerySet)
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/ilids_jstl_gallery.mat')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) and not re_calc_gal_feat \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,querySet=querySet,gallerySet=gallerySet,checkpoint=checkpoint,galfeas_path=gal_savedir)

    print('Evaluate on training set shinpuhkan:')
    dataset=process_shinpuhkan(os.path.join(dataset_dir,'Shinpuhkan/images/'))
    querySet,gallerySet=[],[]
    for img,pid,cid in dataset:
        if cid in [0,1,2,3,4,5,6,7]:
            querySet.append((img,pid,cid))
        if cid in [8,9,10,11,12,13,14,15]:
            gallerySet.append((img,pid,cid))
    query_iter=load_query_or_gallery_iter(querySet)
    gallery_iter=load_query_or_gallery_iter(gallerySet)
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/shinpuhkan_jstl_gallery.mat')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) and not re_calc_gal_feat \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,querySet=querySet,gallerySet=gallerySet,checkpoint=checkpoint,galfeas_path=gal_savedir)