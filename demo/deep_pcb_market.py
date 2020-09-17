'''
Rank-1:89.40% Rank-5:95.87% Rank-10:97.21% Rank-20:98.25% Rank-100:99.32% 
mAP:72.66
耗时1.98小时
'''

from initial import *
from deep.data_manager import Market1501
from deep.data_loader import load_dataset,default_train_transforms,default_test_transforms
from deep.models.ResNet import ResNet50_PCB
from deep.train import train,setup_seed
from deep.test import test
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint,get_rest_params
import torchvision.transforms as T
import torch as pt
import torch.nn as nn

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_PCB.tar')

    market1501=Market1501(dataset_dir)
    market1501.print_info()

    default_train_transforms[0]=T.Resize((384,128))
    default_test_transforms[0]=T.Resize((384,128))
    train_iter,query_iter,gallery_iter=load_dataset(market1501,32,32, \
        train_transforms=default_train_transforms,test_transforms=default_test_transforms)
    
    net=ResNet50_PCB(len(set(list(zip(*market1501.trainSet))[1])))

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