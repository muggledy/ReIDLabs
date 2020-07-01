import torch as pt
import torch.nn as nn
import torchvision
import os.path
import sys
sys.path.append(os.path.dirname(__file__))
from utils import FlattenLayer,Norm1DLayer,HorizontalMaxPool2d

class ResNet50_Classify(nn.Module):
    '''最简单的基于ResNet50的分类网络，适用ID损失，但是注意最后一层并未做SoftMax'''
    def __init__(self,num_ids):
        super(ResNet50_Classify,self).__init__()
        self.train_mode=True #所有模型都要有该参数
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

class ResNet50_Classify_Metric(nn.Module):
    '''ResNet50_Classify的改造版，有少许不同，既适用ID损失，也适用度量损失（譬如三元组损失等度量学习方法），或者两者兼具'''
    def __init__(self,num_ids,loss={'softmax','metric'}): #如果只使用metric，那么num_ids参数无需提供
        super(ResNet50_Classify_Metric,self).__init__()
        self.train_mode=True
        self.loss=loss
        resnet50=torchvision.models.resnet50(pretrained=True)
        self.base=nn.Sequential(
            *(list(resnet50.children())[:-1]),
            FlattenLayer(),
        )
        if 'softmax' in self.loss:
            self.classifier=nn.Linear(2048,num_ids)

    def forward(self,X):
        f=self.base(X)
        if self.train_mode:
            if self.loss=={'softmax'}:
                return self.classifier(f)
            elif self.loss=={'metric'}:
                return f
            elif self.loss=={'softmax','metric'}:
                return self.classifier(f),f
        else:
            return f

class ResNet50_Aligned(nn.Module):
    '''AlignedReID: Surpassing Human-Level Performance in Person Re-Identification
       (arXiv:1711.08184 [cs.CV])
                                                DMLI
                         BN & ReLU & horizon max ⇃
           Resnet50       ┌——————————————————→ lf <triHard loss>
       imgs —————→ f —————|      (N,C)       (N,c,h)
               (N,C,H,W)  └—————→ gf <triHard loss>
                          glob avg | FC layer
                                   ↓
                                  gf1 (N,K)
                               <id loss>
       '''
    def __init__(self,num_ids):
        super(ResNet50_Aligned,self).__init__()
        self.train_mode=True
        resnet50=torchvision.models.resnet50(pretrained=True)
        self.base=nn.Sequential(*(list(resnet50.children())[:-2])) #主干，输出(batch_size,2048,h,w)，后接两个分支
        self.bn=nn.BatchNorm2d(2048) #第一个参数是batchnorm层输入（batch_size,channel,h,w）的通道数channel(2048)
        self.conv1x1=nn.Conv2d(2048,128,kernel_size=1,stride=1,padding=0,bias=True) #可选，加上主要是为了降低特征图
                                                                                    #通道数，加速训练（上图并未标注）
        self.classifier=nn.Linear(2048,num_ids)

    def forward(self,X):
        f=self.base(X)
        gf=nn.AvgPool2d((f.size()[-2:]))(f) #全局分支，全局平均池化，输出(batch_size,channel,1,1)
        gf=FlattenLayer()(gf) #删除空维，使形状为(batch_size,channel)，此处channel其实就是2048，
                              #后接HardTri损失。这也是测试阶段的网络输出
        if self.train_mode:
            gf1=self.classifier(gf) #全局分支的另一个分叉，后接softmax分类损失
            lf=nn.Sequential(self.bn,nn.ReLU(),HorizontalMaxPool2d(),self.conv1x1)(f).squeeze() #局部分
                                                                                    #支，后接TriHard损失（不
                                                                                    #同于全局分支，局部分支
                                                                                    #上的距离矩阵根据DMLI求解）
            lf = lf / pt.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt() #重要！否则出现nan
            return gf1,(gf,lf) #输出形状分别为(batch_size,num_ids), (batch_size,2048), (batch_size,128,h)
        else:
            return gf

if __name__=='__main__':
    model=ResNet50_Aligned(751)
    imgs=pt.randn(32,3,256,128)
    gf1,(gf,lf)=model(imgs)
    print(gf1.shape,gf.shape,lf.shape)
    print(gf1)