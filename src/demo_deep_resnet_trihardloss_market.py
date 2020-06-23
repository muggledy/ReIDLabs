'''
Rank-1:85.54% Rank-5:94.51% Rank-10:96.56% Rank-20:97.74% Rank-100:99.38% 
mAP:69.66
训练用时1.21小时，参数设置如下：
used_losses={'softmax','metric'}, margin=0.3, num_instances=4, lr=0.0003, 
num_epochs=150, batch_size=32, weight_decay=5e-04, step_size=60, gamma=0.1
'''

from code.deep.data_loader import load_Market1501
from code.deep.models.ResNet import ResNet50_Classify_Metric
from code.deep.sampler import RandomIdSampler
from code.deep.train import train,setup_seed
from code.deep.test import test
from code.deep.loss import TripletHardLoss
from code.deep.eval_metric import eval_market1501,eval_cmc_map
from code.deep.models.utils import CheckPoint
from functools import partial
import torch as pt
import torch.nn as nn
import os.path

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(__file__),'../images/Market-1501-v15.09.15/')
    used_losses={'softmax','metric'} #可修改值：{'metric'} or {'softmax'} or {'softmax','metric'}

    margin=0.3
    losses=[]
    losses_name=[]
    cpflag=''
    if 'softmax' in used_losses:
        closs=nn.CrossEntropyLoss()
        losses.append(closs)
        losses_name.append('idLoss')
        cpflag+='C'
    if 'metric' in used_losses:
        mloss=TripletHardLoss(margin)
        losses.append(mloss)
        losses_name.append('triHardLoss')
        cpflag+='M'

    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_Classify_Metric(%s).tar'%cpflag)

    sampler=partial(RandomIdSampler,num_instances=4)
    train_iter,query_iter,gallery_iter,market1501 \
        =load_Market1501(dataset_dir,32,32,sampler=sampler) #注意batch_size必须是num_instances的整数倍
    market1501.print_info()
    net=ResNet50_Classify_Metric(len(market1501.trainPids),loss=used_losses)

    lr,num_epochs=0.0003,150
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=60,gamma=0.1)
    
    train(net,train_iter,losses,optimizer,num_epochs,scheduler, \
        checkpoint=checkpoint,losses_name=losses_name,coeffis=None)
    test(net,query_iter,gallery_iter,eval_cmc_map)