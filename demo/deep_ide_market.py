'''
The first implemented deep reid method with Rank-1 81.24% and mAP 63.83
(Rank-1:81.24% Rank-5:93.05% Rank-10:95.34% Rank-20:96.62% Rank-100:99.02%)
参数设置：lr,num_epochs,weight_decay,step_size(StepLR),gamma,train_batchsize
=0.0003,60,5e-04,20,0.1,32，总需1小时15分（实验1）
2020/6/20
（实验2）Replace softmax loss with oim loss, I got
Rank-1:87.02% Rank-5:94.98% Rank-10:97.15% Rank-20:97.95% Rank-100:99.05% 
mAP:69.71
参数设置：lr,scalar,momentum,num_epochs,weight_decay,step_size(StepLR),gamma,
train_batchsize=0.0003,30,0.5,60,5e-04,20,0.1,32，总需1小时23分
2020/9/18
在OIM基础上，不改变任何参数，使用DataParallel在工作站双2080ti上跑，结果稍低：
Rank-1:86.43% Rank-5:95.07% Rank-10:96.73% Rank-20:97.89% Rank-100:99.11%
mAP:68.57，用时1小时30分钟
我发现单纯地调大train_batchsize（128）并不会带来提升，反而会下降非常多（79%），
该观察是否正确？或许还要改变其他训练参数，譬如增大num_epochs？
（实验3）使用带CBAM的resnet50，测试结果：
Rank-1:87.05% Rank-5:95.31% Rank-10:97.03% Rank-20:98.31% Rank-100:99.61% 
mAP:70.01
毫无提升，用时1小时30分钟。我看知乎是建议直接使用SENet系列，并且是插入到BasicBlock
中，这改变了resnet的基本块结构，导致无法加载预训练参数，所以就不尝试了
（实验4）标签平滑是一个非常有效的技巧，普通交叉熵损失结果（为了减少训练时间，我的
num_epochs设得很小，用时38分钟，lr=0.00035,num_epochs=30）为：
Rank-1:81.12% Rank-5:92.37% Rank-10:94.80% Rank-20:96.79% Rank-100:99.08% 
mAP:61.41
使用平滑交叉熵损失，其他参数保持不变，结果：
Rank-1:85.51% Rank-5:94.54% Rank-10:96.32% Rank-20:97.62% Rank-100:99.32% 
mAP:68.72
注：非常坑的是，在实验2的基础上，我降低了PyTorch版本1.1.0和cudatoolkit版本10.0，
（目的是使用APEX）之前是1.5.0和10.2，结果变低了：
Rank-1:86.19% Rank-5:94.95% Rank-10:96.62% Rank-20:97.77% Rank-100:98.90% 
mAP:67.70
紧接着，我又使用了APEX混合精度测试（opt_level="O1"，原本70%的显卡占用下降到45%）：
Rank-1:87.77% Rank-5:95.64% Rank-10:97.12% Rank-20:98.10% Rank-100:99.23% 
mAP:70.92（用时1小时23分）
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
from deep.loss import OIMLoss,CrossEntropyLabelSmooth
import deep.models.attention.CBAM as CBAM
# from functools import partial

if __name__=='__main__': #话说为什么这部分代码一定要放在__main__块中？好像是多进程加载数据DataLoader的缘故
                         #且仅限于Windows，https://pytorch.apachecn.org/docs/1.2/data.html
    #可在此处设置os.environ['CUDA_VISIBLE_DEVICES']，如'0,1'
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_Classify.tar') #允许随时中断训练进程，但下面使用OIM损失时应尽量一气呵成，因为LUT属于
                                             #模型外参数
    market1501=Market1501(dataset_dir)
    market1501.print_info()
    train_iter,query_iter,gallery_iter=load_dataset(market1501,32,32) #前一个32是训练批次大小，后面一个是测试批次大小，后一个无需修改
    
    num_classes=len(set(list(zip(*market1501.trainSet))[1])) #训练集的行人ID数量
    net=ResNet50_Classify(num_classes,oim=True,backbone=None) #backbone值可以替换成CBAM.resnet50()，带注意力的resnet50版本

    # loss=nn.CrossEntropyLoss() #普通分类交叉熵损失
    # loss=CrossEntropyLabelSmooth(num_classes) #标签平滑损失
    loss=OIMLoss(2048,num_classes,scalar=30,momentum=0.5,device=None) #see in https://github.com/Cysu/open-reid/blob/master/examples/oim_loss.py
    # lr,num_epochs=0.00035,30
    lr,num_epochs=0.0003,60
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04) #权重衰减（正则化）用于应对过拟合
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1) #（steplr等间隔）学习率衰减，
                                                                             #参考：https://zhuanlan.zhihu.com/p/93624972
                                                                             #https://zhuanlan.zhihu.com/p/62585696
    # scheduler=pt.optim.lr_scheduler.MultiStepLR(optimizer,[17,25],gamma=0.1)
    #https://blog.csdn.net/guls999/article/details/85695409
    # optimizer=pt.optim.Adam([{'params':net.parameters(),'initial_lr':lr}],lr=lr,weight_decay=5e-04)
    # scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1,last_epoch=14)
    # scheduler=partial(pt.optim.lr_scheduler.StepLR,optimizer,step_size=10,gamma=0.1)
    #要在重新训练时恢复上次的学习率，像上面仅使用last_epoch是不起作用的，已解决：额外保存和重载optimizer.state_dict()和
    #scheduler.state_dict()，https://www.zhihu.com/question/67209417/answer/250909765
    
    #train参数device可以是None，此时仅仅使用单卡训练（0号），device设为'DP'，会使用全部多卡训练，也可以传递一个数字列表，使用指定的多个卡，
    #但是要注意修改OIM的设备为列表第一项数值，device若设为数字，表示使用指定设备进行单卡训练，此时需要相应修改OIM的设备，取相同数值即可
    #若中途中断训练，是否使用DataParallel也是随时可以修改
    train(net,train_iter,(loss,),optimizer,num_epochs,scheduler,checkpoint=checkpoint,device=None,use_amp=True) #即使你只想利用checkpoint做test，
                                                                             #也必须先执行一下train，由于epoch已达最大，所以实际并不会进行训练
                                                                             #这仅仅是为了完成加载模型参数这一步骤。当然你也可以手动执行加载
                                                                             #net.load_state_dict(...)
    gal_savedir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/market1501_resnetIDE_gallery.mat')
    test(net,query_iter,gallery_iter,eval_cmc_map,save_galFea=gal_savedir,re_rank=False)

    #展示匹配结果（最好放单独文件执行或者取消最开始固定的随机种子，否则每次都会展示相同几幅图片）
    query_dir=os.path.join(dataset_dir,'./query')
    gallery_dir=os.path.join(dataset_dir,'./bounding_box_test')
    plot_match(net,query_dir,gallery_dir,checkpoint,gal_savedir,re_rank=False)