'''
Rank-1:89.01% Rank-5:96.64% Rank-10:97.71% Rank-20:98.60% Rank-100:99.52%
mAP:74.48，用时：1.48小时
参数设置：margin=0.3, num_instances=4, lr=0.0002, num_epochs=200, 
batch_size=32, weight_decay=5e-04, step_size=150, gamma=0.1（测试仅使用了全
局特征，且没有使用任何tricks。理论上使用局部特征测试效果会更好，但我测试时却降低了？）
做了重排后，结果为：
Rank-1:92.13% Rank-5:95.78% Rank-10:96.91% Rank-20:98.04% Rank-100:99.47%
mAP:88.32
参考作者源码：https://github.com/michuanhaohao/AlignedReID
注意，由于分类损失难以下降，目前我已修改了模型，参考BNNeck，在分类层前面加了个BN
层，训练参数俱保持不变，结果：
Rank-1:89.96% Rank-5:96.23% Rank-10:97.65% Rank-20:98.43% Rank-100:99.70%
mAP:75.04，也没啥变化，但重排后mAP提升较大：
Rank-1:92.55% Rank-5:95.93% Rank-10:97.06% Rank-20:97.95% Rank-100:99.64%
mAP:90.20
'''

from initial import *
from deep.data_manager import Market1501
from deep.data_loader import load_dataset
from deep.models.ResNet import ResNet50_Aligned
from deep.sampler import RandomIdSampler
from deep.train import train,setup_seed
from deep.test import test,cosine_dist_T,euc_dist_T
from deep.loss import AlignedTriLoss
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint,calc_dist_DMLI
from functools import partial
import torch as pt
import torch.nn as nn
pt.multiprocessing.set_sharing_strategy('file_system')

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_Aligned.tar')

    sampler=partial(RandomIdSampler,num_instances=4)
    market1501=Market1501(dataset_dir)
    market1501.print_info()
    train_iter,query_iter,gallery_iter=load_dataset(market1501,32,32,sampler=sampler)

    net=ResNet50_Aligned(len(set(list(zip(*market1501.trainSet))[1])))
    margin=0.3
    losses=(nn.CrossEntropyLoss(),AlignedTriLoss(margin))

    lr,num_epochs=0.0002,200
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.1)
    
    train(net,train_iter,losses,optimizer,num_epochs,scheduler,checkpoint=checkpoint, \
        losses_name=['softmaxLoss','globTriHardLoss','localTriHardLoss'],coeffis=None)#,out_loss_map=[[(0,),(0,)],[(1,),(1,)]])
    
    save_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/market1501_resnetAligned_gallery.mat')
    re_calc_gal_fea=True
    # net.aligned=True #aligned
    test(net,query_iter,save_dir if not re_calc_gal_fea and os.path.exists(save_dir) else \
        gallery_iter,eval_cmc_map,
        # calc_dist_funcs=[cosine_dist_T,calc_dist_DMLI], #aligned
        # calc_dist_funcs=euc_dist_T, #test result: rank-1 88.75% by euc_dist but 89.37% by cosine dist
        save_galFea=save_dir,re_rank=False)