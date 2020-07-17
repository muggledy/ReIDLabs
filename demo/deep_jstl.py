'''
无监督模型迁移，在多个无关数据集上训练，在目标数据集上测试。目前来看效果很差，譬如在
cuhk01,cuhk03,market1501数据集上训练，在viper上测试，结果rank1只有17.25%，mAP为21.70
总epoches为60，step_size=40
用时4.9小时
https://github.com/Cysu/dgd_person_reid
'''

from initial import *
from deep.models.ResNet import ResNet56_jstl
from deep.train import train,setup_seed
from deep.test import test
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint
from deep.data_loader import load_dataset,load_train_iter,load_query_or_gallery_iter
from deep.data_manager import MixDataSets,load_query_gallery,Ana716,Ana78,process_cuhk01,process_cuhk03
from deep.plot_match import plot_match
import torch as pt
import torch.nn as nn

if __name__=='__main__':
    # setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(__file__),'../images/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet56_jstl(no_viper).tar')

    mixdatasets=MixDataSets(('cuhk01',os.path.join(dataset_dir,'CUHK01')), \
        ('cuhk03',os.path.join(dataset_dir,'cuhk03_images/detected')), \
        ('market1501',os.path.join(dataset_dir,'Market-1501-v15.09.15')))

    batch_size=32
    train_iter=load_train_iter(mixdatasets.dataset)

    net=ResNet56_jstl(len(set(list(zip(*mixdatasets.dataset))[1])))
    loss=nn.CrossEntropyLoss()
    lr,num_epochs=0.0003,60
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
    train(net,train_iter,(loss,),optimizer,num_epochs,scheduler,checkpoint=checkpoint)

    print('Test on target dataset VIPeR:')
    query_gallery=load_query_gallery(os.path.join(dataset_dir,'./VIPeR.v1.0/cam_a'), \
        os.path.join(dataset_dir,'./VIPeR.v1.0/cam_b'),query_num=None,analyse=Ana716(r'(\d{,3})_'))
    query_imgs,query_pids,_=list(zip(*(query_gallery.querySet)))
    query_gallery.querySet=list(zip(query_imgs,query_pids,[0]*632)) #对VIPeR数据集必须手动设置摄像头ID
    gallery_imgs,gallery_pids,_=list(zip(*(query_gallery.gallerySet)))
    query_gallery.gallerySet=list(zip(gallery_imgs,gallery_pids,[1]*632))
    query_iter,gallery_iter=load_dataset(query_gallery,test_batch_size=batch_size,notrain=True)

    gal_savedir=os.path.join(os.path.dirname(__file__),'../data/viper_jstl_gallery.npz')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,query_gallery=query_gallery,checkpoint=checkpoint,galfeas_path=gal_savedir)

    #在训练集market1501、cuhk01、03上测试，可想而知rank-1和mAP都会非常高
    print('Test on training set Market1501:')
    query_dir=os.path.join(dataset_dir,'./Market-1501-v15.09.15/query')
    gallery_dir=os.path.join(dataset_dir,'./Market-1501-v15.09.15/bounding_box_test')
    gal_savedir=os.path.join(os.path.dirname(__file__),'../data/market1501_jstl_gallery.npz')
    query_gallery=load_query_gallery(query_dir,gallery_dir,query_num=None,analyse=Ana78(r'([-\d]+)_c(\d)'))
    query_iter,gallery_iter=load_dataset(query_gallery,test_batch_size=batch_size,notrain=True)
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    plot_match(net,query_gallery=query_gallery,checkpoint=checkpoint,galfeas_path=gal_savedir)

    print('Test on training set CUHK01:')
    dataset=process_cuhk01(os.path.join(dataset_dir,'./CUHK01'))
    querySet,gallerySet=[],[]
    for img,pid,cid in dataset:
        if cid==0:
            querySet.append((img,pid,cid))
        elif cid==1:
            gallerySet.append((img,pid,cid))
    query_iter=load_query_or_gallery_iter(querySet)
    gallery_iter=load_query_or_gallery_iter(gallerySet)
    gal_savedir=os.path.join(os.path.dirname(__file__),'../data/cuhk01_jstl_gallery.npz')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    class obj: pass
    query_gallery=obj()
    setattr(query_gallery,'querySet',querySet)
    setattr(query_gallery,'gallerySet',gallerySet)
    plot_match(net,query_gallery=query_gallery,checkpoint=checkpoint,galfeas_path=gal_savedir)

    print('Test on training set CUHK03:')
    dataset=process_cuhk03(os.path.join(dataset_dir,'./cuhk03_images/detected'))
    querySet,gallerySet=[],[]
    for img,pid,cid in dataset:
        if cid in [0,2,4,6,8]:
            querySet.append((img,pid,cid))
        elif cid in [1,3,5,7,9]:
            gallerySet.append((img,pid,cid))
    query_iter=load_query_or_gallery_iter(querySet)
    gallery_iter=load_query_or_gallery_iter(gallerySet)
    gal_savedir=os.path.join(os.path.dirname(__file__),'../data/cuhk03_jstl_gallery.npz')
    test(net,query_iter,gal_savedir if os.path.exists(gal_savedir) \
        else gallery_iter,eval_cmc_map,save_galFea=gal_savedir)
    setattr(query_gallery,'querySet',querySet)
    setattr(query_gallery,'gallerySet',gallerySet)
    plot_match(net,query_gallery=query_gallery,checkpoint=checkpoint,galfeas_path=gal_savedir)