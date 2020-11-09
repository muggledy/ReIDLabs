'''
Softmax Loss + Ring Loss，结果：
Rank-1:85.78% Rank-5:94.54% Rank-10:96.85% Rank-20:97.92% Rank-100:99.35% 
mAP:70.89
单纯使用Softmax Loss，结果：
Rank-1:84.32% Rank-5:93.38% Rank-10:95.90% Rank-20:97.48% Rank-100:98.96% 
mAP:67.62
'''

from initial import *
from deep.data_manager import Market1501
from deep.data_loader import load_dataset
from deep.models.ResNet import ResNet50_Classify_Metric
from deep.sampler import RandomIdSampler
from deep.train import train,setup_seed
from deep.test import test
from deep.loss import RingLoss
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint
import torch as pt

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), \
        '../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_Softmax_Ring_Loss.tar')
    
    market1501=Market1501(dataset_dir)
    market1501.print_info()
    train_iter,query_iter,gallery_iter=load_dataset(market1501,64,32)

    net=ResNet50_Classify_Metric(loss={'softmax','metric'}, \
        num_ids=len(set(list(zip(*market1501.trainSet))[1]))) #分类模型，'metric'选项仅仅是为了输出样本特征，以供RingLoss使用，而非用于度量损失
    softmax_loss=pt.nn.CrossEntropyLoss()
    ring_loss=RingLoss()
    alpha=0.1
    optimizer=pt.optim.Adam([{
        'params':net.parameters(),'lr':0.0003,'name':'resnet_softmax'
    },{
        'params':ring_loss.parameters(),'lr':0.5/alpha,'name':'ring_loss'
    }],weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)

    num_epochs=100
    train(net,train_iter,(softmax_loss,ring_loss),optimizer,num_epochs,scheduler, \
        checkpoint=checkpoint,losses_name=('SoftmaxLoss','RingLoss'), \
        coeffis=(1,alpha),use_amp=True,device='DP', \
        if_tensorboard=True,tensorboard_subdir='deep_test_ringloss', \
        use_pcids=['P','*'], \
        # if_visdom=True, \
        query_iter=query_iter,gallery_iter=gallery_iter,evaluate=eval_cmc_map) #每一个epoch都计算测试集精度极耗费时间，用于选择最佳epoch训练次数
                                                              #原则上不能使用测试集，而应在训练集中划分验证集，但是此处不好划分验证集
    # test(net,query_iter,gallery_iter,eval_cmc_map)