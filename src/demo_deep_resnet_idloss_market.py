'''
The first implemented deep reid method with Rank-1 81.24% and mAP 63.83
(Rank-1:81.24% Rank-5:93.05% Rank-10:95.34% Rank-20:96.62% Rank-100:99.02%)
按照罗浩的设置（lr,num_epochs,weight_decay,step_size,gamma,train_batchsize,
test_batchsize=0.0003,60,5e-04,20,0.1,32,32），总需1小时15分，本
代码思路基本和罗浩提供的代码一致（除了缺少一个随机crop的transform，加不加无所谓）
2020/6/20
'''

from code.deep.data_manager import Market1501
from code.deep.data_loader import load_dataset
from code.deep.models.ResNet import ResNet50_Classify
from code.deep.train import train,setup_seed
from code.deep.test import test
from code.deep.eval_metric import eval_market1501,eval_cmc_map
from code.deep.models.utils import CheckPoint
import os.path
import torch as pt
import torch.nn as nn

if __name__=='__main__': #为什么这部分代码一定要放在__main__块中？
    setup_seed(0) #尽管设置了种子，但是每次结果可能仍有稍许不同，大概零点几个百分点区别，如果去掉此行
                  #，每次结果则会有很大不同，我可能哪里设置的不对？
    dataset_dir=os.path.join(os.path.dirname(__file__),'../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_Classify.tar') #允许随时中断训练进程

    market1501=Market1501(dataset_dir)
    market1501.print_info()
    train_iter,query_iter,gallery_iter=load_dataset(market1501,32,32)
    
    net=ResNet50_Classify(len(market1501.trainPids))

    loss=nn.CrossEntropyLoss()
    lr,num_epochs=0.0003,10
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=6,gamma=0.1) #学习率衰减，参考：https://zhuanlan.zhihu.com/p/93624972
                                                                            #注意checkpoint无法继续上次的step_size衰减过程，如果权重衰
                                                                            #减地厉害，请下次执行的时候人为修改为上次结束时的学习率，以
                                                                            #降低影响
    train(net,train_iter,(loss,),optimizer,num_epochs,scheduler,checkpoint=checkpoint) #即使你只想利用checkpoint做test，也必须先执行一下
                                                                                       #train，由于epoch已达最大，所以实际并不会进行训练
    test(net,query_iter,gallery_iter,eval_cmc_map)
