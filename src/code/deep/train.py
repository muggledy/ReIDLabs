import torch as pt
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from tools import measure_time

def setup_seed(seed):
    print('Use random seed(%d)'%seed)
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic=True

@measure_time
def train(net,train_iter,loss,optimizer,epochs,scheduler=None,device=None,checkpoint=None):
    flag='cuda' if pt.cuda.is_available() else 'cpu'
    if device is None:
        device=pt.device(flag)
    net=net.to(device)
    print('Now training on %s...'%device)
    if flag=='cuda':
        cudnn.benchmark=True
        net=nn.DataParallel(net).cuda() #允许多卡训练
    else:
        print('Currently using CPU(and GPU is highly recommended)')
    if checkpoint.loaded:
        net.load_state_dict(checkpoint.states_info_etc['state']) #注意必须在执行net=DataParallel(net)之后加载参数
    net.train()
    net.train_mode=True
    if checkpoint is not None and checkpoint.loaded:
        start_epoch=checkpoint.states_info_etc['epoch']+1
    else:
        start_epoch=0
    all_batches_num=len(train_iter)
    for epoch in range(start_epoch,epochs):
        for i,(batchImgs,pids,cids) in enumerate(train_iter):
            batchImgs,pids=batchImgs.to(device),pids.to(device) #目前用不到cids摄像头信息
            out=net(batchImgs)
            l=loss(out,pids)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if i==0 or (i+1)%20==0:
                print('epoch=%d, batch=[%d/%d], loss=%f, lr=%f'%(epoch+1,i+1,all_batches_num,l, \
                    optimizer.param_groups[0]["lr"] if scheduler is None else scheduler.get_last_lr()[0]))
        if scheduler is not None:
            scheduler.step()
        if checkpoint is not None:
            checkpoint.save({'state':net.state_dict(),'epoch':epoch})
    print('Train OVER!')
