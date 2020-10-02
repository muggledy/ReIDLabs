# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
comes from https://github.com/Xiangyu-CAS/AICity2020-VOC-ReID/blob/7c453723e6e9179d175772921f93441cfa621dc1/lib/solver/lr_scheduler.py
"""
from bisect import bisect_right
import torch
import math


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=10, #the period of warmup
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters #self.last_epoch increases from 0 to self.warmup_iters and alpha 0 to 1 linearly
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha #warmup_factor=self.warmup_factor+(1-self.warnup_factor)*alpha
                                                            #so warmup_factor increases from self.warmup_factor to 1 linearly in warmup stage
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


'''
Bag of Tricks for Image Classification with Convolutional Neural Networks
'''
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        warmup_epochs=10,
        eta_min=1e-7,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs - 1
        self.eta_min=eta_min
        self.warmup_epochs = warmup_epochs
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = [base_lr * (self.last_epoch+1) / (self.warmup_epochs + 1e-32) for base_lr in self.base_lrs]
        else:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]
        return lr


class CosineStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        step_epochs=2,
        gamma=0.3,
        eta_min=0,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs
        self.eta_min=eta_min
        self.step_epochs = step_epochs
        self.gamma = gamma
        self.last_cosine_lr = 0
        super(CosineStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.max_epochs - self.step_epochs:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch) / (self.max_epochs - self.step_epochs))) / 2
                    for base_lr in self.base_lrs]
            self.last_cosine_lr = lr
        else:
            lr = [self.gamma ** (self.step_epochs - self.max_epochs + self.last_epoch + 1) * base_lr for base_lr in self.last_cosine_lr]

        return lr


class CyclicCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 cycle_epoch,
                 cycle_decay=0.7,
                 last_epoch=-1):
        self.cycle_decay = cycle_decay
        self.cycle_epoch = cycle_epoch
        self.cur_count = 0
        super(CyclicCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.cur_count = (self.last_epoch + 1) // self.cycle_epoch
        decay = self.cycle_decay ** self.cur_count
        return [base_lr * decay *
         (1 + math.cos(math.pi * (self.last_epoch % self.cycle_epoch) / self.cycle_epoch)) / 2
         for base_lr in self.base_lrs]


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from functools import partial
    from collections import defaultdict
    epochs=100
    LR = 0.00035
    schedulers = {
        'WarmupLR':partial(WarmupMultiStepLR,warmup_iters=10,warmup_factor=0.1,milestones=[40,70],gamma=0.1),
        'WarmupCosineLR':partial(WarmupCosineLR,max_epochs=40),
        'CosineStepLR':partial(CosineStepLR,max_epochs=30),
        'CyclicCosineLR':partial(CyclicCosineLR,cycle_epoch=30),
        'StepLR':partial(torch.optim.lr_scheduler.StepLR,step_size=10,gamma=0.5),
        'MultiStepLR':partial(torch.optim.lr_scheduler.MultiStepLR,milestones=[20,80],gamma=0.9),
        'ExponentialLR':partial(torch.optim.lr_scheduler.ExponentialLR,gamma=0.9),
        'CosineAnnealingLR':partial(torch.optim.lr_scheduler.CosineAnnealingLR,T_max=20),
        'LambdaLR':partial(torch.optim.lr_scheduler.LambdaLR,lr_lambda=lambda epoch:math.sin(epoch)/(epoch+1)+0.5), #https://www.jianshu.com/p/9643cba47655
    }
    n=len(schedulers)
    lr_lists=defaultdict(list)

    for name,scheduler in schedulers.items():
        model = torch.nn.Linear(6,6)
        optimizer = torch.optim.Adam(model.parameters(),lr = LR)
        scheduler = scheduler(optimizer)
        for epoch in range(epochs):
            lr_lists[name].append(optimizer.state_dict()['param_groups'][0]['lr'])
            scheduler.step()
    
    h=int(math.sqrt(n))
    w=int(math.ceil(n/h))
    if h==1:
        w=n
    for i,(name,lr_list) in enumerate(lr_lists.items()):
        plt.subplot(h,w,i+1)
        plt.plot(range(epochs),lr_list,color = 'r')
        plt.xlabel(name)
        plt.minorticks_on()
        plt.grid(which='both')
    plt.show()