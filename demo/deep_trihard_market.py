'''
Rank-1:85.54% Rank-5:94.51% Rank-10:96.56% Rank-20:97.74% Rank-100:99.38% 
mAP:69.66
训练用时1.21小时，参数设置如下：
used_losses={'softmax','metric'}, margin=0.3, num_instances=4, lr=0.0003, 
num_epochs=150, batch_size=32, weight_decay=5e-04, step_size=60, gamma=0.1
如果只使用TriHard损失：
Rank-1:83.17% Rank-5:93.94% Rank-10:96.14% Rank-20:97.86% Rank-100:99.29% 
mAP:65.52，用时25分钟（num_epochs=30,num_instances=4,num_paddings=8,
step_size=12,lr=0.0003,batch_size=32），在此基础上，设置use_amp=True,
batch_size=64，结果：
Rank-1:83.76% Rank-5:94.48% Rank-10:96.59% Rank-20:97.86% Rank-100:99.23% 
mAP:66.63
由于参数我是自己随便调的，论文给出的TriHard损失结果是：rank-1为83.8%，mAP为68.0
关于MSML的测试：
直接使用在imagenet上预训练的resnet50测试结果：
Rank-1:8.82% Rank-5:17.73% Rank-10:23.63% Rank-20:31.38% Rank-100:53.38% 
mAP:2.40
使用MSML（引入“相对距离”版本），设置batch_size为16（每个行人挑4张），大概50个
epoch后结果（MultiStepLR[20,50]）：
Rank-1:14.93% Rank-5:37.05% Rank-10:48.99% Rank-20:61.58% Rank-100:86.58% 
mAP:6.45
随着epoch上升，结果也在缓慢提升，大概150个epoch，结果：
Rank-1:42.28% Rank-5:69.48% Rank-10:78.89% Rank-20:86.94% Rank-100:96.41% 
mAP:23.63
大概300个epoch：
Rank-1:64.52% Rank-5:84.71% Rank-10:89.96% Rank-20:93.50% Rank-100:98.34% 
mAP:43.06，耗时漫长，效果很差，实在百无一用（一直到900个epoch，也不过70%）
如果使用原版MSML，150个epoch后（其他训练参数不变），rank-1只提升了1个点：
Rank-1:9.62% Rank-5:28.03% Rank-10:38.78% Rank-20:51.01% Rank-100:79.31% 
mAP:4.09
250个epoch，rank-1才不过17%：
Rank-1:17.87% Rank-5:42.22% Rank-10:55.11% Rank-20:67.07% Rank-100:88.95% 
mAP:8.02
越大的batch_size，结果越差，譬如我使用32(batch_size)是越训越差（50个epoch就降至
3%），这也
容易理解，因为每次只会从整个批次中挑选出一个三元组对，剩下的就都不管（至少我认为
其作用有限），batch_size越大，每一epoch实际参与训练的样本越少，导致训练陷于一个
极差的局部解
可惜作者始终没有公布源码，为什么他论文中批次大小设为128也能正常工作？是不是得要训
上千个epoch？在market1501上能达到85%？疑惑
'''

from initial import *
from deep.data_manager import Market1501
from deep.data_loader import load_dataset
from deep.models.ResNet import ResNet50_Classify_Metric,\
    ResNet50_Classify,ResNet56_Classify
from deep.sampler import RandomIdSampler,RandomIdSampler2
from deep.train import train,setup_seed
from deep.test import test
from deep.loss import TripletHardLoss,MSMLoss
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint
from functools import partial
import torch as pt
import torch.nn as nn
from deep.plot_match import plot_match

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/Market-1501-v15.09.15/')
    ### Test with TriHard Loss + ID Loss
    used_losses={'metric'} #可修改值：{'metric'} or {'softmax'} or {'softmax','metric'}
    use_amp=False

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
    
    ## Test with MSML（结果异常，我无法解决）
    # checkpoint=CheckPoint()
    # checkpoint.load('ResNet50_MSML.tar')
    # sampler=partial(RandomIdSampler2,num_instances=4,num_paddings=None, \
    #     statistics=True)
    # market1501=Market1501(dataset_dir)
    # market1501.print_info()
    # train_iter,query_iter,gallery_iter \
    #     =load_dataset(market1501,16,32,sampler=sampler)
    # losses=(MSMLoss(0.3),)
    # losses_name=('MSMLoss',)
    # use_amp=False
    # net=ResNet50_Classify_Metric(loss={'metric'})
    # lr,num_epochs=0.00035,1000
    # optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    # scheduler=pt.optim.lr_scheduler.MultiStepLR(optimizer,[100,150],gamma=0.1)
    ###
    
    train(net,train_iter,losses,optimizer,num_epochs,scheduler, \
        checkpoint=checkpoint,losses_name=losses_name,coeffis=None,use_amp=use_amp)
    save_gal_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), \
        '../data/market1501_resnetMSML_gallery.mat')
    test(net,query_iter,gallery_iter,eval_cmc_map,save_galFea=save_gal_path)

    plot_match(net,query_gallery=market1501,galfeas_path=save_gal_path)