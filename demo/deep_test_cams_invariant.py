'''
关于“摄像头不变特征”的测试，观察其在跨域实验中的性能
（实验一）Market1501(source domain)——>DukeMTMC(target domain)
(Market1501)Rank-1:87.02% Rank-5:95.55% Rank-10:97.15% Rank-20:98.16% Rank-100:99.29% 
mAP:69.12
(DukeMTMC)Rank-1:20.06% Rank-5:34.11% Rank-10:40.35% Rank-20:47.35% Rank-100:63.73% 
mAP:8.70
（实验二）在实验一原有IDE模型的基础上，添加摄像头判别器，接在特征提取器也就是resnet50后面，通
过对抗训练学习提取摄像头不变特征的能力，然后再直接应用到目标域，观察是否有提升
images from different cameras ——> resnet50 ——> features ——> ID Loss
                                                  └——> discriminate different cameras
为了简便，效仿2015_DANN，在resnet50即特征提取器和摄像头判别器之间添加GRL梯度反转层以替代对抗
学习（的最大最小训练策略）
设特征提取器为G，行人id分类器为C，摄像头判别器为D，总体优化目标就是：
    min C(G(X))+lambda*D(R(G(X)))
其中R表示梯度反转层
(Market1501)Rank-1:85.96% Rank-5:95.28% Rank-10:96.91% Rank-20:98.13% Rank-100:99.41% 
mAP:68.55
(DukeMTMC)Rank-1:20.96% Rank-5:36.45% Rank-10:43.09% Rank-20:50.04% Rank-100:66.61% 
mAP:9.66
好像也没什么提升，可以再改改参数，我一共调了两次参数，第一次结果更低，虽然这一次结果也不高，仅
仅是直接迁移到Duke上的结果变高了一点
2020/11/15
注（用Market1501一样的参数训练DukeMTMC数据集）：
Rank-1:80.21% Rank-5:90.57% Rank-10:93.22% Rank-20:95.38% Rank-100:97.98% 
mAP:63.16
'''

from initial import *
from deep.data_manager import Market1501,DukeMTMC
from deep.data_loader import load_dataset
from deep.models.ResNet import ResNet50_Classify, \
    ResNet50_Classify_CamInvariant
from deep.train import train,setup_seed
from deep.test import test,extract_feats
from deep.eval_metric import eval_market1501,eval_cmc_map
from deep.models.utils import CheckPoint
import torch as pt
import torch.nn as nn
from deep.plot_match import plot_match
from deep.loss import OIMLoss
from zoo.plot import plot_dataset
from deep.data_loader import DataLoader,T,reidDataset, \
    default_train_transforms

if __name__=='__main__':
    setup_seed(0)

    ###在market1501上训练IDE网络
    checkpoint=CheckPoint()
    # checkpoint.load('CamsInvariant_Classify_Market.tar') #-（实验一）
    checkpoint.load('CamsInvariant_Classify_Market2.tar') #+（实验二）
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/')
    market1501=Market1501(os.path.join(dataset_dir,'Market-1501-v15.09.15'))
    market1501.print_info()
    market_train_iter,market_query_iter,market_gallery_iter=load_dataset(market1501,32,32)
    market_num_train_pids=len(set(list(zip(*market1501.trainSet))[1]))
    market_num_train_cids=len(set(list(zip(*market1501.trainSet))[2]))
    num_epochs=60
    # net=ResNet50_Classify(market_num_train_pids,oim=True) #-
    net=ResNet50_Classify_CamInvariant(market_num_train_pids,oim=True, \
        cam_nums=market_num_train_cids,alpha=None, \
        count=num_epochs*len(market_train_iter)) #+
    pids_loss=OIMLoss(2048,market_num_train_pids,scalar=30,momentum=0.5,device=None)
    cids_loss=nn.CrossEntropyLoss() #+
    lr=0.0003
    # optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04) #-
    lambd=1 #+
    optimizer=pt.optim.Adam([{
        'params':net.base.parameters(),'lr':lr,'name':'backbone'
    },{
        'params':net.camera_discrimination.parameters(),'lr':lr/lambd,'name':'discrimination'
    }],weight_decay=5e-04) #+
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)
    # train(net,market_train_iter,(pids_loss,),optimizer,num_epochs,scheduler, \
    #     checkpoint=checkpoint,device=None,use_amp=True) #-
    train(net,market_train_iter,(pids_loss,cids_loss),optimizer,num_epochs,scheduler, \
        coeffis=(1,lambd), \
        checkpoint=checkpoint,device=None,use_pcids=['P','C'],use_amp=True) #+
    market_gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)), \
        '../data/CamsInvariant_Classify_Market_Gal.mat')
    re_calc_gal=True
    test(net,market_query_iter,market_gal_savedir if not re_calc_gal \
        and os.path.exists(market_gal_savedir) else market_gallery_iter,eval_cmc_map, \
        save_galFea=market_gal_savedir,re_rank=False)
    
    ###直接在duke上测试
    dukemtmc=DukeMTMC(os.path.join(dataset_dir,'DukeMTMC'))
    dukemtmc.print_info()
    duke_train_iter,duke_query_iter,duke_gallery_iter=load_dataset(dukemtmc,32,32)
    duke_gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)), \
        '../data/CamsInvariant_Classify_Duke_Gal.mat')
    test(net,duke_query_iter,duke_gal_savedir if not re_calc_gal \
        and os.path.exists(duke_gal_savedir) else duke_gallery_iter, \
        eval_cmc_map,save_galFea=duke_gal_savedir,re_rank=False)

    ###观察market1501和dukemtmc两个数据集（训练集）空间分布
    load_full_trainset=lambda trainset,batch_size,num_workers: \
        DataLoader(reidDataset(trainset,T.Compose(default_train_transforms)), \
        batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False)
    market_train_feats=extract_feats(net,load_full_trainset(market1501.trainSet,32,4))
    duke_train_feats=extract_feats(net,load_full_trainset(dukemtmc.trainSet,32,4))
    market_camids=np.array(list(zip(*market1501.trainSet))[2])
    duke_camids=np.array(list(zip(*dukemtmc.trainSet))[2])+np.max(market_camids)+1
    plot_dataset(market_train_feats,list(zip(*market1501.trainSet))[1],market_camids, \
        dim=3,only_colour_cameras=True) #观察market1501下不同摄像头数据分布
    plot_dataset(duke_train_feats,list(zip(*dukemtmc.trainSet))[1],duke_camids, \
        dim=3,only_colour_cameras=True) #观察dukemtmc下不同摄像头数据分布
    plot_dataset(np.concatenate((market_train_feats,duke_train_feats),1), \
        np.concatenate((market_camids,duke_camids)),[0]*len(market_camids)+ \
        [1]*len(duke_camids),dim=3,only_colour_cameras=True) #观察market1501和dukemtmc两个数据集分布差异
    #注：就观察来看，在market1501训练集上训练后，market1501训练集各个摄像头下图像特征分布融合地
    #比较深，而未训练的dukemtmc训练集下各个摄像头特征分布差别则较大，此外，market1501和dukemtmc
    #两个数据集之间的分布差异也比较大

    '''
    ###附：在duke上训练IDE网络
    checkpoint.load('CamsInvariant_Classify_Duke.tar')
    duke_num_train_pids=len(set(list(zip(*dukemtmc.trainSet))[1]))
    net=ResNet50_Classify(duke_num_train_pids,oim=True)
    loss=OIMLoss(2048,duke_num_train_pids,scalar=30,momentum=0.5,device=None)
    lr,num_epochs=0.0003,60
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)
    train(net,duke_train_iter,(loss,),optimizer,num_epochs,scheduler, \
        checkpoint=checkpoint,device=None,use_amp=True)
    re_calc_gal=False
    duke_gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)), \
        '../data/CamsInvariant_Classify_Duke_Gal2.mat')
    test(net,duke_query_iter,duke_gal_savedir if not re_calc_gal and \
        os.path.exists(duke_gal_savedir) else duke_gallery_iter,eval_cmc_map, \
        save_galFea=duke_gal_savedir,re_rank=False)
    plot_match(net,querySet=dukemtmc.querySet,gallerySet=dukemtmc.gallerySet, \
        checkpoint=checkpoint,galfeas_path=duke_gal_savedir,re_rank=False)
    '''