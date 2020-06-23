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
def train(net,train_iter,losses,optimizer,epochs,scheduler=None,coeffis=None,device=None,checkpoint=None,**kwargs):
    '''注意，对于losses参数，即使只有一个损失，也要写为(loss,)或者[loss]即序列形式。总损失loss
       =coeffis[0]*losses[0](net_out,targets)+coeffis[1]*losses[1](net_out,targets)+...'''
    flag='cuda' if pt.cuda.is_available() else 'cpu'
    if device is None:
        device=pt.device(flag)
    net=net.to(device)
    print('Now training on %s...'%device)
    if flag=='cuda':
        cudnn.benchmark=True
        net=nn.DataParallel(net).cuda() #允许多卡训练
    else:
        print('Currently using CPU(but GPU is highly recommended)')
    if checkpoint is not None and checkpoint.loaded:
        net.load_state_dict(checkpoint.states_info_etc['state']) #注意必须在执行net=DataParallel(net)之后加载参数
        start_epoch=checkpoint.states_info_etc['epoch']+1
    else:
        start_epoch=0
    net.train()
    net.train_mode=True
    
    all_batches_num=len(train_iter)
    losses_num=len(losses)
    losses_name=kwargs.get('losses_name')
    if losses_name is not None:
        if len(losses_name)!=losses_num:
            raise ValueError('Losses_name\'s length(%d) must be equal to losses\' length(%d)!' \
                %(len(losses_name),losses_num))
    else:
        losses_name=['loss%d'%(i+1) for i in range(losses_num)]
    if coeffis is None:
        coeffis=[1]*len(losses)
    elif len(coeffis)!=losses_num:
        raise ValueError('Coeffis\'s length(%d) must be equal to losses\' length(%d)!'%(len(coeffis),losses_num))
    for epoch in range(start_epoch,epochs):
        for batch_ind,(batchImgs,pids,cids) in enumerate(train_iter):
            batchImgs,pids=batchImgs.to(device),pids.to(device) #目前用不到cids摄像头信息
            out=net(batchImgs)
            if not isinstance(out,(list,tuple)):
                out=[out]
            if len(out)!=losses_num:
                raise ValueError('The num(%d) of net\'s out must be equal to losses\' num(%d)!'%(len(out),losses_num))
            L=0
            Llist=[]
            for losses_ind,loss in enumerate(losses):
                subL=loss(out[losses_ind],pids)
                Llist.append(subL)
                L+=(coeffis[losses_ind]*subL)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            if batch_ind==0 or (batch_ind+1)%20==0:
                print('epoch=%d, batch=[%d/%d], %s, lr=%f'%(epoch+1,batch_ind+1,all_batches_num, \
                    'loss=%f'%L if losses_num==1 else ('loss(all)=%f, '%L)+ \
                    (', '.join(['%s=%f'%(e,Llist[i]) for i,e in enumerate(losses_name)])), \
                    optimizer.param_groups[0]["lr"] if scheduler is None else scheduler.get_last_lr()[0]))
        if scheduler is not None:
            scheduler.step()
        if checkpoint is not None:
            checkpoint.save({'state':net.state_dict(),'epoch':epoch})
    print('Train OVER!')
