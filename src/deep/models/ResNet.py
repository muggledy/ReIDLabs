import torch as pt
import torch.nn as nn
import torchvision
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from deep.models.utils import FlattenLayer,Norm1DLayer,HorizontalPool2d,seek_ks_3m,_weights_init,LambdaLayer
from torchvision.models.resnet import Bottleneck
import torch.nn.functional as F
import copy

class ResNet50_Classify(nn.Module): #https://blog.csdn.net/qq_31347869/article/details/100566719
    '''最简单的基于ResNet50的分类网络，适用ID损失，但是注意最后一层并未做SoftMax'''
    def __init__(self,num_ids):
        super(ResNet50_Classify,self).__init__()
        self.train_mode=True #所有模型都要有该参数
        resnet50=torchvision.models.resnet50(pretrained=True) #此处要删除原ResNet50的最后一层，因为最后一层是1000分
                                                              #类，不适用于当前任务（譬如Market1601训练集为751分类）
                                                              ##https://github.com/akamaster/pytorch_resnet_cifar10
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
        self.aligned=False
        resnet50=torchvision.models.resnet50(pretrained=True)
        self.base=nn.Sequential(*(list(resnet50.children())[:-2])) #主干，输出(batch_size,2048,h,w)，后接两个分支
        self.bn=nn.BatchNorm2d(2048) #第一个参数是batchnorm层输入（batch_size,channel,h,w）的通道数channel(2048)
        self.conv1x1=nn.Conv2d(2048,128,kernel_size=1,stride=1,padding=0,bias=True) #可选，加上主要是为了降低特征图
                                                                                    #通道数，加速训练（上图并未标注）
        self.classifier=nn.Linear(2048,num_ids)
                                     #卷积尺寸计算：⌊ (H-K_h+2P_h)/S_h + 1 ⌋ and ⌊ (W-K_w+2P_w)/S_w + 1 ⌋
                                     #where K is kernel(K_h,K_w) and P is padding(P_h,P_w) and S is stride(S_h,S_w)
    def forward(self,X):
        f=self.base(X)
        gf=nn.AvgPool2d((f.size()[-2:]))(f) #全局分支，全局平均池化，输出(batch_size,channel,1,1)
        gf=FlattenLayer()(gf) #删除空维，使形状为(batch_size,channel)，此处channel其实就是2048，
                              #后接HardTri损失。这也是测试阶段的网络输出。也可以用squeeze()代替
        local_stream=nn.Sequential(self.bn,nn.ReLU(),HorizontalPool2d(pool_type='max'),self.conv1x1)
        norm=lambda X: X / pt.pow(X,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        if self.train_mode:
            gf1=self.classifier(gf) #全局分支的另一个分叉，后接softmax分类损失
            lf=local_stream(f).squeeze() #局部分支，后接TriHard损失（不同于全局分支，局部分支上的距离矩阵根据DMLI求解）
            lf = norm(lf) #重要！否则出现nan
            return gf1,(gf,lf) #输出形状分别为(batch_size,num_ids), (batch_size,2048), (batch_size,128,h)
        else:
            if self.aligned:
                lf=norm(local_stream(f).squeeze())
                return gf,lf
            return gf

class ResNet50_PCB(nn.Module): #PCB's baseline, refer to 
                               #https://github.com/syfafterzy/PCB_RPP_for_reID/blob/master/reid/models/resnet.py
    def __init__(self,num_ids):
        super(ResNet50_PCB,self).__init__()
        self.train_mode=True
        resnet50=torchvision.models.resnet50(pretrained=True) #this is an extremely long dream, dream of parallel world
        layer4_1=Bottleneck(1024,512,downsample=nn.Sequential(nn.Conv2d(1024,2048,1,bias=False),nn.BatchNorm2d(2048))) #
                                                           #https://blog.csdn.net/qq_34108714/article/details/90169562
        pretrained_layer4_1_state=resnet50.layer4[0].state_dict()
        delete_keys=['conv2.weight','downsample.1.weight'] #这两层更改了，所以预训练的参数无法起作用
        layer4_1.state_dict().update({k:v for k,v in pretrained_layer4_1_state.items() if k not in delete_keys})
        resnet50.layer4[0]=layer4_1 #修改resnet50的layer4第一个block第二个卷积步长为1，以及下采样层stride为1，这样可以保持layer3
                                    #的输出特征图尺寸不变，更好地提取特征，目前很多论文都是这样处理的
        self.base=nn.Sequential(*list(resnet50.children())[:-2]) #举例来说，假设网络输入图像尺寸为256x128，未修改前输出为8x4，
                                                                 #修改后变为16x8，因为原本的下采样比例是1/2
        self.outsize=6 #经过水平池化后的FM尺寸为6x1，即划分6个水平条
        self.ks=None
        self.conv1x1s=[]
        self.classifiers=[]
        for i in range(self.outsize): #所有用到的（带参数的）网络层必须在init中预先声明，避免不必要的错误
            setattr(self,'conv1x1_%d'%i,nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0,bias=True))
            self.conv1x1s.append(getattr(self,'conv1x1_%d'%i))
            setattr(self,'classifier_%d'%i,nn.Linear(256,num_ids))
            self.classifiers.append(getattr(self,'classifier_%d'%i))

    def forward(self,X):
        X=self.base(X)
        if self.ks is None:
            self.ks=seek_ks_3m(X.size(2),self.outsize)[:2] #此处X.size(2)为24，按照3m原则，计算会得到
                                                           #核尺寸为4，步长也为4。水平池化后特征图为6x1
        X=HorizontalPool2d(*self.ks,pool_type='avg')(X)
        ret,fs=[],[]
        for i in range(self.outsize):
            fs.append(self.conv1x1s[i](X[...,[i],:]).squeeze())
        if self.train_mode:
            for i in range(self.outsize):
                ret.append(self.classifiers[i](fs[i])) #每个水平条后接一个1x1的卷积+全连接层
            return ret #用以接6个softmax损失
        else:
            return pt.cat(fs,dim=1)

class ResNet50_MGN(nn.Module): #copy from https://github.com/seathiefwang/MGN-pytorch/blob/master/model/mgn.py
    def __init__(self, num_classes, pool='avg', feats=256):
        super(ResNet50_MGN, self).__init__()
        self.train_mode=True
        resnet = torchvision.models.resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        
        if pool == 'max':
            pool2d = nn.MaxPool2d
        elif pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        
        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)
        
        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        if self.train_mode:
            return fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
        else:
            predict = pt.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
            return predict

if __name__=='__main__':
    pass