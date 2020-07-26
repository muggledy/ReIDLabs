'''
The first implemented deep reid method with Rank-1 81.24% and mAP 63.83
(Rank-1:81.24% Rank-5:93.05% Rank-10:95.34% Rank-20:96.62% Rank-100:99.02%)
参数设置：lr,num_epochs,weight_decay,step_size(StepLR),gamma,train_batchsize,
test_batchsize=0.0003,60,5e-04,20,0.1,32,32，总需1小时15分
2020/6/20
'''

from initial import *
from deep.data_manager import Market1501
from deep.data_loader import load_dataset
from deep.models.ResNet import ResNet50_Classify
from deep.train import train,setup_seed
from deep.test import test
from deep.eval_metric import eval_market1501,eval_cmc_map
from deep.models.utils import CheckPoint
import torch as pt
import torch.nn as nn
from deep.plot_match import plot_match
# from functools import partial

if __name__=='__main__': #话说为什么这部分代码一定要放在__main__块中？好像是多线程的缘故
    setup_seed(0) #尽管设置了种子，但是每次结果可能仍有稍许不同，大概零点几个百分点区别，如果去掉此行
                  #，每次结果则会有很大不同，我可能哪里设置的不对？
    dataset_dir=os.path.join(os.path.dirname(__file__),'../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_Classify.tar') #允许随时中断训练进程

    market1501=Market1501(dataset_dir)
    market1501.print_info()
    train_iter,query_iter,gallery_iter=load_dataset(market1501,32,32)
    
    net=ResNet50_Classify(len(set(list(zip(*market1501.trainSet))[1]))) #传入训练集的ID数量

    loss=nn.CrossEntropyLoss()
    lr,num_epochs=0.00035,30
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04) #权重衰减（正则化）用于应对过拟合
    # scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1) #（steplr等间隔）学习率衰减，
                                                                             #参考：https://zhuanlan.zhihu.com/p/93624972
                                                                             #https://zhuanlan.zhihu.com/p/62585696
    scheduler=pt.optim.lr_scheduler.MultiStepLR(optimizer,[17,25],gamma=0.1)
    #https://blog.csdn.net/guls999/article/details/85695409
    # optimizer=pt.optim.Adam([{'params':net.parameters(),'initial_lr':lr}],lr=lr,weight_decay=5e-04)
    # scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1,last_epoch=14)
    # scheduler=partial(pt.optim.lr_scheduler.StepLR,optimizer,step_size=10,gamma=0.1)
    #要在重新训练时恢复上次的学习率，像上面仅使用last_epoch是不起作用的，已解决：额外保存和重载optimizer.state_dict()和
    #scheduler.state_dict()，https://www.zhihu.com/question/67209417/answer/250909765
    
    train(net,train_iter,(loss,),optimizer,num_epochs,scheduler,checkpoint=checkpoint) #即使你只想利用checkpoint做test，也必须先执行一下
                                                                                       #train，由于epoch已达最大，所以实际并不会进行训练
                                                                                       #这仅仅是为了完成加载模型参数这一步骤
    gal_savedir=os.path.join(os.path.dirname(__file__),'../data/market1501_resnetIDE_gallery.mat')
    test(net,query_iter,gallery_iter,eval_cmc_map,save_galFea=gal_savedir,re_rank=False)

    #展示匹配结果（最好放单独文件执行或者取消最开始固定的随机种子，否则每次都会展示相同几幅图片）
    query_dir=os.path.join(dataset_dir,'./query')
    gallery_dir=os.path.join(dataset_dir,'./bounding_box_test')
    plot_match(net,query_dir,gallery_dir,checkpoint,gal_savedir,re_rank=False)