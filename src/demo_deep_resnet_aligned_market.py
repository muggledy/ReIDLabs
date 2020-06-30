'''
Rank-1:89.07% Rank-5:95.96% Rank-10:97.45% Rank-20:98.43% Rank-100:99.52% 
mAP:74.33
参数设置：margin=0.3, num_instances=4, lr=0.0002, num_epochs=206, 
batch_size=32, weight_decay=5e-04, step_size=150, gamma=0.1（测试仅使用了全
局特征，且没有使用任何tricks）
用时：1.48小时
参考作者源码：https://github.com/michuanhaohao/AlignedReID
'''

from code.deep.data_loader import load_Market1501
from code.deep.models.ResNet import ResNet50_Aligned
from code.deep.sampler import RandomIdSampler
from code.deep.train import train,setup_seed
from code.deep.test import test
from code.deep.loss import AlignedTriLoss
from code.deep.eval_metric import eval_market1501,eval_cmc_map
from code.deep.models.utils import CheckPoint
from functools import partial
import torch as pt
import torch.nn as nn
import os.path

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(__file__),'../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_Aligned.tar')

    sampler=partial(RandomIdSampler,num_instances=4)
    train_iter,query_iter,gallery_iter,market1501 \
        =load_Market1501(dataset_dir,32,32,sampler=sampler)
    market1501.print_info()

    net=ResNet50_Aligned(len(market1501.trainPids))
    margin=0.3
    losses=(nn.CrossEntropyLoss(),AlignedTriLoss(margin))

    lr,num_epochs=0.0002,206
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.1)
    
    train(net,train_iter,losses,optimizer,num_epochs,scheduler, \
        checkpoint=checkpoint,losses_name= \
        ['softmaxLoss','globTriHardLoss','localTriHardLoss'],coeffis=None)
    test(net,query_iter,gallery_iter,eval_cmc_map)
