from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../'))
from deep.uda.ARN.mmd import mix_rbf_mmd2
from deep.models.utils import euc_dist
from deep.loss import TripletHardLoss,LabelSmoothLoss

class ReconstructionLoss(nn.Module): #图像误差重构，如果是重构原始图像的话，倒是可以考虑SSIM（一种图像质量评估方法）
    def __init__(self, dist_metric='L1'):
        super(ReconstructionLoss, self).__init__()
        self.dist_metric = dist_metric
        
    def forward(self, re_img, gt_img):
        if self.dist_metric == 'L1':
            p = 1
        elif self.dist_metric == 'L2':
            p = 2
        b,c,h,w = gt_img.size()
        loss = torch.dist(re_img, gt_img, p=p) / (b*h*w) #https://pytorch.org/docs/master/generated/torch.norm.html#torch-norm
                                                         #为何除以(b*h*w)而非(b*c*h*w)
        return loss

class MMDLoss(nn.Module):
    def __init__(self, base=1.0, sigma_list=[1, 2, 10]):
        super(MMDLoss, self).__init__()
        # sigma for MMD
        #         self.sigma_list = sigma_list
        self.base = base
        self.sigma_list = sigma_list
        self.sigma_list = [sigma / self.base for sigma in self.sigma_list]

    def forward(self, Target, Source):
        Target = Target.view(Target.size()[0], -1)
        Source = Source.view(Source.size()[0], -1)
        mmd2_D = mix_rbf_mmd2(Target, Source, self.sigma_list)
        mmd2_D = F.relu(mmd2_D)
        mmd2_D = torch.sqrt(mmd2_D)
        return mmd2_D

# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=2):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, predict, gt):
#         predict = predict.view(predict.size()[0], -1)
#         batch, dim = predict.size()
#         loss = 0.0
#         for i in range(batch):
#             for j in range(i, batch):
#                 if gt[i] == gt[j]:
#                     label = 1
#                 else:
#                     label = 0
#                 dist = torch.dist(predict[i], predict[j], p=2) ** 2 / dim #为何除以dim？
#                 loss += label * dist + (1 - label) * F.relu(self.margin - dist)
#         loss = 2 * loss / (batch * (batch - 1))
#         return loss

class ContrastiveLoss(nn.Module):
    '''图像Ia和Ib之间的对比损失
           loss = y * dist(Ia,Ib) + (1-y) * max(0, margin - dist(Ia,Ib))
       当Ia和Ib是同一行人时，y=1，否则y=0
    '''
    def __init__(self, margin=0.3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets): #inputs: (batch_size, feat_dim), targets: (num_classes,)
                                        #我参照三元组损失重写了ARN源码所实现的对比损失函数，也就是上面注释掉
                                        #的代码，在ARN实验中，若仅使用分类损失和对比损失，20个batch耗时9.35
                                        #秒，在替换成重写后的对比损失后，时耗立即降低到3.97秒，巨大的提升
        n = inputs.size(0)
        dist=euc_dist(inputs.t()).sqrt() #对称阵
        mask_p = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_n=mask_p==False
        
        triuinds=torch.triu_indices(n,n)
        mask_p[triuinds[0],triuinds[1]]=False #mask_p中为True的位置(i,j)代表图像Ii和Ij是同一行人
        mask_n[triuinds[0],triuinds[1]]=False #mask_n中为True的位置(i,j)代表图像Ii和Ij是不同行人

        dist_ap, dist_an = dist[mask_p], dist[mask_n] #找出批次中所有同一行人对以及不同行人对
        return (torch.sum(dist_ap)+torch.sum(torch.max( \
            self.margin-dist_an,torch.zeros_like(dist_an)))) \
            /(len(dist_ap)+len(dist_an))

class DifferenceLoss(nn.Module):
    def forward(self, feature1, feature2):
        feature1 = feature1.view(-1)
        feature2 = feature2.view(-1)
        loss = torch.dot(feature1, feature2)
        return loss

if __name__=='__main__':
    pass