import torch as pt
import torch.nn as nn
import torchvision
import os.path
import sys
sys.path.append(os.path.dirname(__file__))
from utils import FlattenLayer,Norm1DLayer

class ResNet50_Classify(nn.Module):
    '''最简单的基于ResNet50的分类网络，但是注意最后一层并未做SoftMax'''
    def __init__(self,num_ids):
        super(ResNet50_Classify,self).__init__()
        self.train_mode=True
        resnet50=torchvision.models.resnet50(pretrained=True) #此处要删除原ResNet50的最后一层，因为最后一层是1000分
                                                              #类，不适用于当前任务（譬如Market1601训练集为751分类）
        self.base=nn.Sequential(
            *(list(resnet50.children())[:-1]),
            FlattenLayer(), #打印resnet50可知倒数第二层输出形状为(batch_size,2048,1,1)
                            #，需要做一下Flatten删除最后两个空维以供后面全连接层的输入
            # Norm1DLayer() #可选，标准化特征向量，使长度为1
        ) #基干网（提取特征）
        self.classifier=nn.Linear(2048,num_ids) #分类网，输出层节点数等于分类数，全连接层

    def forward(self,X): #输入数据的形状为(batch_size,channels,height,width)
        f=self.base(X)
        if self.train_mode:
            return self.classifier(f)
        else:
            return f

if __name__=='__main__':
    model=ResNet50_Classify(751)
    t=pt.arange(24*4).view(2,3,4,4).to(pt.float32)
    model.train_mode=False
    out=model(t)
    print(out.shape)
    print((out[0]*out[0]).sum())
    from utils import print_net_size
    print_net_size(model)