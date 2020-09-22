import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../'))
from zoo.tools import Crawler

'''
简易版resnet56实现（网络输出维度64维），原打算用于CAMEL那篇论文实验部分，可结果实在太低
要是能在ImageNet上先训练一遍应该会有提升
comes from https://github.com/Brother-Lee/CIFAR10_ResNet56_v2/tree/master/pytorch
但该实现存在错误，目前已纠正
also see in https://blog.csdn.net/briblue/article/details/84325722
'''

__all__ = ['resnet56']


def conv3x3(in_c_out, out_c_out, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_c_out, out_c_out, kernel_size=3, stride=stride,padding=1)


def conv1x1(in_c_out, out_c_out, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_c_out, out_c_out, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(c_in, c_out, stride)
        self.downsample = downsample
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = conv3x3(c_out, c_out)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x) #由于使用3x3卷积间接使残差（out）达到了下采样效果（除非使用padding），所以为了能和输入
                                          #x相加，需要对x也做下采样，使尺寸一致

        out += identity
        return out

class Bottleneck(nn.Module):
    expansion = 4 #block输出通道数相对block输入通道数的扩张比例

    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = conv1x1(c_in, c_out)

        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = conv3x3(c_out, c_out, stride)

        self.bn3 = nn.BatchNorm2d(c_out)
        self.conv3 = conv1x1(c_out, c_out * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)


        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10): #基本的结构块block有两种，一是BasicBlock，二是Bottleneck，两者的主要区别是，
                                                       #前者是由两个3x3卷积层构成，通道数前后不变，后者首先使用1x1卷积对block输入通
                                                       #道数进行了缩小，然后使用一个3x3卷积，最后再使用一个1x1卷积对通道数进行了恢复
                                                       #（以使block输出通道数和输入保持一致）或扩张（这是“瓶颈”一词的由来）
        super(ResNet, self).__init__()
        self.in_planes=16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0]) #原论文包括5个stage，其中一个conv1，四个layer，每个layer都由多个
                                                             #基本块block串成
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.batch_norm = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.1)


    def _make_layer(self, block, planes, blocks,stride=1):
        downsample=None
        if self.in_planes!=planes*block.expansion or stride!=1: #判断是否需要进行下采样
            downsample = conv1x1(self.in_planes, planes * block.expansion, stride)
        layers = []
        
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes=planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)

        return x

#论文中resnet18,34使用了BasicBlock，resnet50,101,152使用了Bottleneck


def resnet56(pretrained=False,model_state_file=None):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [9, 9, 9]) #源码若替换成Bottleneck是存在错误的

    if pretrained:
        print('Get pretrained(on CIFRA10) resnet56',end='')
        if model_state_file==None:
            model_state_file=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../data/model-247-0.9379.pth'))
        if not os.path.exists(model_state_file):
            print(' from Internet(default:model-247-0.9379.pth)...')
            link='https://link.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcjdORVNsRGhsRXRjRFc1TzNZci0weDhLZ2s/ZT15VnNyT0M=.mp3'
            crawler=Crawler()
            crawler.get(link,use_proxy=True)
            crawler.save(model_state_file,auto_detect_suffix=False)
        else:
            print(' from local %s'%model_state_file)
        model_state=torch.load(model_state_file)
        model.load_state_dict(model_state['state_dict'])
    return model

if __name__=='__main__':
    a=resnet56(True)
    b=torch.randn(1,3,256,128)
    c=a(b)
    print(c.shape)
