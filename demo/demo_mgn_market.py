'''
多粒度网络（MGN）
重排后结果（参数：lr=0002,num_epochs=80,step_size=60,gamma=0.1，另外所有损失
权重系数全部置为1）：
Rank-1:94.45% Rank-5:97.27% Rank-10:97.86% Rank-20:98.75% Rank-100:99.70%
mAP:92.35
用时1.62小时
设置num_epochs=100，结果基本不变：
Rank-1:94.89% Rank-5:97.62% Rank-10:98.16% Rank-20:98.93% Rank-100:99.67%
mAP:92.79
如果不做重排，结果为：
Rank-1:93.68% Rank-5:97.89% Rank-10:98.66% Rank-20:99.29% Rank-100:99.67%
mAP:82.98
注：结果之所以和论文有所不同，是因为我没有完全参照它的训练参数，机器的原因可能很小
'''

from initial import *
from deep.data_manager import Market1501
from deep.sampler import RandomIdSampler
from deep.data_loader import load_dataset, \
    default_train_transforms,default_test_transforms
from deep.models.ResNet import ResNet50_MGN
from deep.loss import TripletHardLoss
from deep.train import train,setup_seed
from deep.test import test
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint
import torchvision.transforms as T
import torch as pt
import torch.nn as nn
from functools import partial

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(__file__),'../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_MGN.tar')

    sampler=partial(RandomIdSampler,num_instances=4)
    market1501=Market1501(dataset_dir)
    market1501.print_info()
    default_train_transforms[0]=T.Resize((384,128))
    default_test_transforms[0]=T.Resize((384,128))
    train_iter,query_iter,gallery_iter=load_dataset(market1501,32,32,sampler=sampler, \
        train_transforms=default_train_transforms,test_transforms=default_test_transforms)

    net=ResNet50_MGN(len(set(list(zip(*market1501.trainSet))[1])))

    softmax_loss=nn.CrossEntropyLoss()
    triplet_loss=TripletHardLoss(margin=0.3)

    losses=(triplet_loss,triplet_loss,triplet_loss,softmax_loss,softmax_loss,softmax_loss, \
        softmax_loss,softmax_loss,softmax_loss,softmax_loss,softmax_loss)

    lr,num_epochs=0.0002,100
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=60,gamma=0.1)

    train(net,train_iter,losses,optimizer,num_epochs,scheduler,checkpoint=checkpoint, \
        coeffis=None) #coeffis设为None，表示所有子损失融合权重都为1

    save_dir=os.path.join(os.path.dirname(__file__),'../data/market1501_resnetMGN_gallery.mat')
    re_calc_gal_fea=True
    test(net,query_iter,save_dir if not re_calc_gal_fea and os.path.exists(save_dir) else \
        gallery_iter,eval_cmc_map,save_galFea=save_dir,re_rank=False)