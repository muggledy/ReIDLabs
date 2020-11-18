'''
Rank-1:84.12% Rank-5:94.09% Rank-10:96.47% Rank-20:97.54% Rank-100:99.14%
mAP:66.02，耗时46min
'''

from initial import *
from deep.data_manager import Market1501
from deep.data_loader import load_dataset
from deep.models.ResNet import ResNet50_Classify_Metric
from deep.sampler import RandomIdSampler
from deep.train import train,setup_seed
from deep.test import test
from deep.loss import TripletHardLoss,CenterLoss
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint
from functools import partial
import torch as pt
pt.multiprocessing.set_sharing_strategy('file_system')

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), \
        '../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_Triplet_Center_Loss.tar')
    
    sampler=partial(RandomIdSampler,num_instances=4)
    market1501=Market1501(dataset_dir)
    market1501.print_info()
    train_iter,query_iter,gallery_iter=load_dataset(market1501,64,32,sampler=sampler)

    net=ResNet50_Classify_Metric(loss={'metric'}) #度量学习模型
    triplet_loss=TripletHardLoss(margin=0.3)
    center_loss=CenterLoss(num_classes=len(set(list(zip(*market1501.trainSet))[1])),feat_dim=2048)
    alpha=0.0001
    optimizer=pt.optim.Adam([{
        'params':net.parameters(),'lr':0.0003,'name':'resnet_metric'
    },{
        'params':center_loss.parameters(),'lr':0.5/alpha,'name':'center_loss'
    }],weight_decay=5e-04) #网络参数学习率设为0.0003，中心损失中参数（记作theta）学习率设为0.5（三元组损失不带参数），
    #总体损失为all_loss=triplet_loss+alpha*center_loss，所以all_loss对theta求偏导时会带上（乘）一个系数也就是alpha，
    #这个系数和损失无关，为了消除该系数影响，所以将theta相关的学习率乘以一个(1./alpha)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)

    num_epochs=100
    train(net,train_iter,(triplet_loss,center_loss),optimizer,num_epochs,scheduler, \
        checkpoint=checkpoint,losses_name=('TriHardLoss','CenterLoss'), \
        # if_visdom=True, \
        coeffis=(1,alpha),use_amp=True,out_loss_map=[[(0,),(0,1)]]) #将网络的输出（网络只有一个输出）分别作为第一
                                                                    #个损失和第二个损失的输入，一共两个损失
    test(net,query_iter,gallery_iter,eval_cmc_map)