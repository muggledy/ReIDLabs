'''
Rank-1:85.54% Rank-5:94.51% Rank-10:96.56% Rank-20:97.74% Rank-100:99.38% 
mAP:69.66
训练用时1.21小时，参数设置如下：
used_losses={'softmax','metric'}, margin=0.3, num_instances=4, lr=0.0003, 
num_epochs=150, batch_size=32, weight_decay=5e-04, step_size=60, gamma=0.1
'''

from initial import *
from deep.data_manager import Market1501
from deep.data_loader import load_dataset
from deep.models.ResNet import ResNet50_Classify_Metric
from deep.sampler import RandomIdSampler,RandomIdSampler2
from deep.train import train,setup_seed
from deep.test import test
from deep.loss import TripletHardLoss
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint
from functools import partial
import torch as pt
import torch.nn as nn

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/Market-1501-v15.09.15/')
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

    sampler=partial(RandomIdSampler2,num_instances=4,num_paddings=8, \
        statistics=True) #如果不想使用num_paddings扩充，直接置为None即可
    market1501=Market1501(dataset_dir)
    market1501.print_info()
    train_iter,query_iter,gallery_iter \
        =load_dataset(market1501,32,32,sampler=sampler) #注意batch_size必须是num_instances的整数倍
    
    net=ResNet50_Classify_Metric(len(set(list(zip(*market1501.trainSet))[1])),loss=used_losses)

    lr,num_epochs=0.0003,30
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=12,gamma=0.1)
    
    train(net,train_iter,losses,optimizer,num_epochs,scheduler, \
        checkpoint=checkpoint,losses_name=losses_name,coeffis=None)
    test(net,query_iter,gallery_iter,eval_cmc_map)