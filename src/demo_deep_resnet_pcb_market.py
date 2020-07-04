'''
Rank-1:89.40% Rank-5:95.87% Rank-10:97.21% Rank-20:98.25% Rank-100:99.32% 
mAP:72.66
耗时1.98小时
'''

from code.deep.data_manager import Market1501
from code.deep.data_loader import load_dataset
from code.deep.models.ResNet import ResNet50_PCB
from code.deep.train import train,setup_seed
from code.deep.test import test
from code.deep.eval_metric import eval_cmc_map
from code.deep.models.utils import CheckPoint,get_rest_params
import torchvision.transforms as T
import os.path
import torch as pt
import torch.nn as nn

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(__file__),'../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_PCB.tar')

    market1501=Market1501(dataset_dir)
    market1501.print_info()

    normalizer=T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
    train_transforms=[T.Resize((384,128)),T.RandomHorizontalFlip(),T.ToTensor(), \
            normalizer]
    test_transforms=[T.Resize((384,128)),T.ToTensor(),normalizer]
    train_iter,query_iter,gallery_iter=load_dataset(market1501,32,32, \
        train_transforms=train_transforms,test_transforms=test_transforms)
    
    net=ResNet50_PCB(len(market1501.trainPids))

    losses=[]
    for i in range(6):
        losses.append(nn.CrossEntropyLoss())
    lr,num_epochs=0.0003,60
    
    optimizer=pt.optim.Adam([{
        'params':get_rest_params(net.base),'lr':lr*1,'name':'base'
    },{
        'params':get_rest_params(net,['base']),'lr':lr,'name':'ide'
    }],weight_decay=5e-04) #https://blog.csdn.net/qq_34914551/article/details/87699317
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
    train(net,train_iter,losses,optimizer,num_epochs,scheduler,checkpoint=checkpoint)
    test(net,query_iter,gallery_iter,eval_cmc_map)