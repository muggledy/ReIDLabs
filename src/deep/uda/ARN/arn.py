from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../'))
from deep.models.utils import FlattenLayer,print_net_size

class Base_Encoder(nn.Module):
    def __init__(self):
        super(Base_Encoder, self).__init__()
        self.base = models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(self.base.children())[:-3])

    def forward(self, input_img):
        return self.base(input_img)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.base = models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(self.base.children())[-3:-2])
    
    def forward(self, input_feature):
        return self.base(input_feature)

class Decoder(nn.Module):
    def __init__(self, ch_list=None):
        super(Decoder, self).__init__()

        def get_layers(in_filters, out_filters, stride=1, out_pad=0):
            layers = [nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3,\
                        stride=stride, padding=1, output_padding=out_pad),
                     nn.BatchNorm2d(out_filters),
                     nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            return layers
        self.block1 = nn.Sequential(
            *get_layers(ch_list[0], ch_list[1], stride=2, out_pad=1),
            *get_layers(ch_list[1], ch_list[1]),
            *get_layers(ch_list[1], ch_list[1])
        )

    def forward(self, input_feature):
        return self.block1(input_feature)

# class Decoder(nn.Module): #我现在更加倾向于作者github是故意给出一个错误的代码，因为这和论文里都不一致了
#     def __init__(self, ch_list=None):
#         super(Decoder, self).__init__()
#         def get_layers(in_filters, out_filters, stride=1, out_pad=0, last=False):
#             layers = [nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3,\
#                         stride=stride, padding=1, output_padding=out_pad),
#                      nn.BatchNorm2d(out_filters)]
#             if last:
#                 layers.append(nn.Tanh())
#             else:
#                 layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
#             return layers

#         def make_blocks(in_ch, out_ch, last=False):
#             block = nn.Sequential(
#                 *get_layers(in_ch, out_ch, stride=2, out_pad=1),
#                 *get_layers(out_ch, out_ch),
#                 *get_layers(out_ch, out_ch, last=last)
#             )
#             return block

#         self.block1 = make_blocks(ch_list[0], ch_list[1])
#         self.block2 = make_blocks(ch_list[1], ch_list[2])
#         self.block3 = make_blocks(ch_list[2], ch_list[3])
#         self.block4 = make_blocks(ch_list[3], ch_list[4])
#         self.block5 = make_blocks(ch_list[4], ch_list[5], last=True)

#     def forward(self, input_feature):
#         feature1 = self.block1(input_feature)
#         feature2 = self.block2(feature1)
#         feature3 = self.block3(feature2)
#         feature4 = self.block4(feature3)
#         feature5 = self.block5(feature4)
#         return feature5

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, BN=False, Drop=True):
        super(Classifier, self).__init__()
        self.layers=[]
        if BN:
            self.layers.append(nn.BatchNorm1d(input_dim))
        if Drop:
            self.layers.append(nn.Dropout(p=0.5 if not isinstance(Drop,float) else Drop))
        self.layers.append(nn.Linear(input_dim, output_dim))
        self.layers=nn.Sequential(*self.layers)

    def forward(self, feature):
        return self.layers(feature)

class AdaptReID_model(nn.Module):
    def __init__(self, classifier_input_dim, classifier_output_dim):
        super(AdaptReID_model, self).__init__()
        self.train_mode=True

        self.encoder_base = Base_Encoder()
        self.encoder_t = Encoder()
        self.encoder_c = Encoder()
        self.encoder_s = Encoder()

        self.avgpool = nn.Sequential(nn.AvgPool2d((8,4)),FlattenLayer()) #作者的意图其实就是做全局平均池化，因为
                                                                         #AvgPool2d的输入特征图空间尺寸就是(8,4)
        self.ch_list = [2048, 1024, 512, 256, 64, 3]
        self.decoder_c = Decoder(ch_list=self.ch_list)
        self.classifier = Classifier(input_dim=classifier_input_dim, output_dim=classifier_output_dim, \
            BN=True,Drop=False) #BNNeck

    def forward(self, image_s=None, image_t=None):
        if self.train_mode:
            feature_s = self.encoder_base(image_s) #(1, 1024, 16, 8)
            feature_s_es = self.encoder_s(feature_s) #(1, 2048, 8, 4)
            feature_s_es_avg = self.avgpool(feature_s_es) #(1, 2048)
            feature_s_ec = self.encoder_c(feature_s) #(1, 2048, 8, 4)
            feature_s_ec_avg = self.avgpool(feature_s_ec) #(1, 2048)
            
            feature_s_ = feature_s_ec + feature_s_es #(1, 2048, 8, 4)
            image_s_ = self.decoder_c(feature_s_) #(1, 3, 256, 128)，论文里明明说是对Base_Encoder所得特征图feature_s（以源域为例）
                                                  #进行分解，但重构时却是对原始图像image_s（应该是feature_s啊）与image_s_
                                                  #（Decoder重构结果）进行误差重构，所以恢复源码中被注释掉的Decoder，使Decoder的输
                                                  #出尺寸为(1, 1024, 16, 8)
            pred_s = self.classifier(feature_s_ec_avg)

            feature_t = self.encoder_base(image_t)
            feature_t_et = self.encoder_t(feature_t)
            feature_t_et_avg = self.avgpool(feature_t_et)
            feature_t_ec = self.encoder_c(feature_t)
            feature_t_ec_avg = self.avgpool(feature_t_ec)
            
            feature_t_ = feature_t_ec + feature_t_et
            image_t_ = self.decoder_c(feature_t_)
            
            return feature_s, feature_t, feature_s_es_avg, feature_s_ec_avg, \
                feature_t_et_avg, feature_t_ec_avg, image_s_, image_t_, pred_s
        else:
            if image_s is not None and image_t is None:
                return self.avgpool(self.encoder_c(self.encoder_base(image_s)))
            elif image_s is None and image_t is not None:
                return self.avgpool(self.encoder_c(self.encoder_base(image_t)))

if __name__=='__main__':
    model=AdaptReID_model(2048,751)
    t=torch.Tensor(1,3,256,128)
    print_net_size(Base_Encoder())
    print_net_size(Encoder())
    model(t,t)