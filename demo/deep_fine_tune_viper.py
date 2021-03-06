'''
对deep_model_trans.py训练的无监督迁移模型在目标测试集viper上做微调（实验1），
观察结果。随机对半划分viper，一半作为训练集，一半作为测试集，使用无监督迁移模
型在测试集上rank-1为26.90%，用训练集微调后，测试集上再测试结果为46.84%，用时34min
在deep_model_trans.py实验2的基础上，测试集结果为33.23%，用训练集微调后，测试
集重新测试结果为51.90%
这个结果有点低了，传统方法一般也能达到55%，甚至有达到99%的，这已经是在大量reid数
据集上训练过迁移的模型，如果不迁移，直接拿一半训练，结果岂不是更低？如何提升？
该问题是否可以归结于一般性的小样本数据集的训练结果较低问题
'''

from initial import *
from deep.models.ResNet import ResNet50_Classify
from deep.train import train,setup_seed
from deep.test import test
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint,get_rest_params
from deep.data_loader import load_train_iter,load_query_or_gallery_iter
from deep.data_manager import process_viper
import torch as pt
import torch.nn as nn
from deep.plot_match import plot_match

class fine_tune_net(ResNet50_Classify): #原始训练的模型输出有3940个节点，现在要在viper上微调，
                                        #输出节点数只有316，在加载预训练参数的时候，注意就不要
                                        #加载最后的分类层了
    def __init__(self,num_ids):
        super(fine_tune_net,self).__init__(4476)
        checkpoint=CheckPoint()
        checkpoint.load('ResNet56_jstl(no_viper).tar')
        if checkpoint.loaded:
            self.load_state_dict(checkpoint.states_info_etc['state'])
        self.classifier=nn.Linear(2048,num_ids)

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/')
    trainSet,querySet,gallerySet=process_viper(os.path.join(dataset_dir,'./VIPeR.v1.0'),split_train_test=True)
    train_iter=load_train_iter(trainSet)
    query_iter=load_query_or_gallery_iter(querySet)
    gallery_iter=load_query_or_gallery_iter(gallerySet)

    batch_size=32
    net=fine_tune_net(len(set(list(zip(*trainSet))[1])))
    test(net,query_iter,gallery_iter,eval_cmc_map)
    
    loss=nn.CrossEntropyLoss()
    lr,num_epochs=0.0002,250
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.1)

    # lr,num_epochs=0.0002,250
    # optimizer=pt.optim.Adam([{
    #     'params':get_rest_params(net.base),'lr':lr*0.1,'name':'base'
    # },{
    #     'params':get_rest_params(net,['base']),'lr':lr,'name':'ide'
    # }],weight_decay=5e-04)
    # scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=130,gamma=0.1)

    train(net,train_iter,(loss,),optimizer,num_epochs,scheduler)
    test(net,query_iter,gallery_iter,eval_cmc_map)

    plot_match(net,querySet=querySet,gallerySet=gallerySet)