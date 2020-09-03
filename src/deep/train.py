import torch as pt
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from zoo.tools import measure_time
# from functools import partial

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
       =coeffis[0]*losses[0](net_out[0],targets)+coeffis[1]*losses[1](net_out[1],targets)+...'''
    if checkpoint is not None and checkpoint.loaded:
        net.load_state_dict(checkpoint.states_info_etc['state'])
        start_epoch=checkpoint.states_info_etc['epoch']+1
        optimizer.load_state_dict(checkpoint.states_info_etc['optimizer']) #必须配合保存重载optimizer，scheduler才起作用
        for state in optimizer.state.values(): #https://blog.csdn.net/weixin_41848012/article/details/105675735
            for k, v in state.items():
                if pt.is_tensor(v):
                    state[k] = v.cuda()
        if checkpoint.states_info_etc.get('scheduler'):
            scheduler.load_state_dict(checkpoint.states_info_etc['scheduler'])
    else:
        start_epoch=0
    # if scheduler is not None and isinstance(scheduler,partial):
    #     scheduler=scheduler(last_epoch=start_epoch-1)
    # if scheduler is not None:
    #     scheduler.step(start_epoch-1)
    checkpoint.states_info_etc=None #在训练MGN模型时，采用PK采样（用于三元组损失），P=8，K=4，事实上此时显存占用已达60%-70%（8G），
    #当我暂停训练，并从checkpoint继续时，得到一条错误：CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
    #隐约感到这是显存不够导致的，我降低了批次大小，令P=2，不再报错，于是定位到此处，在从checkpoint加载模型参数后，将checkpoint置为None
    #（以释放空间）
    checkpoint.loaded=False #与此同时，将loaded标志置为False，下次必须重新加载才行

    flag='cuda' if pt.cuda.is_available() else 'cpu' #如果存在GPU，优先使用GPU
    origin_device=device
    device=pt.device(flag) if device is None else device
    net=net.to(device)
    if flag=='cuda' and origin_device is None: #如果device已经人为给出，即拥有最高优先级。如果没有给出
                                               #为None，才会尝试放到多GPU设备上运行，假设存在GPU设备的话
        cudnn.benchmark=True
        net=nn.DataParallel(net) #允许多卡训练，https://blog.csdn.net/zhjm07054115/article/details/104799661/
        print('Set cudnn.benchmark=True')
        gpu_num=pt.cuda.device_count()
        if gpu_num==1:
            print('Try training on multi-GPU, but you have only one GPU:%d'%pt.cuda.current_device())
        else:
            print('Training on %d GPU devices'%gpu_num)
    else:
        print('Now training on %s...'%device)
    
    net.train()
    if isinstance(net,nn.DataParallel):
        net.module.train_mode=True
    else:
        net.train_mode=True
    
    all_batches_num=len(train_iter)
    losses_num=len(losses)
    losses_name=kwargs.get('losses_name')
    if losses_name is None: #实际losses_name的长度应等于所有损失函数的输出数量的总和，这边只是先赋一个初值，后面可能
                            #还会进行修改，因为有些损失函数的输出不止一个，譬如
                            #AlignedReID的三元组损失有两个输出，这种情况下建议人为给出losses_name，主要是使调用者清
                            #楚，打印时出现的子损失个数大于losses参数长度时不要吃惊
        losses_name=['loss%d'%(i+1) for i in range(losses_num)]
    if coeffis is None: #coeffis长度默认和losses_name保持一致，默认值全为1
        coeffis=[1]*len(losses_name)
    coeffis_lossesname_flag=True
    
    for epoch in range(start_epoch,epochs):
        for batch_ind,(batchImgs,pids,cids) in enumerate(train_iter):
            batchImgs,pids=batchImgs.to(device),pids.to(device) #目前用不到cids摄像头信息
            # print([False for i in net.parameters() if not i.is_cuda]) #查看是否有网络参数不在cuda上
            out=net(batchImgs)
            if not isinstance(out,(list,tuple)):
                out=[out]
            if len(out)!=losses_num: #假设网络有n个输出，那么必须有相应的n个损失（这个n为参数losses长度）
                raise ValueError('The num(%d) of net\'s out must be equal to losses\' length(%d)!'%(len(out),losses_num))
            # L=0 #
            Llist=[]
            for losses_ind,loss in enumerate(losses):
                subL=loss(out[losses_ind],pids.to(pt.int64))
                if isinstance(subL,(list,tuple)): ## 要求损失函数的输出要么是单个0维tensor，要么是多个0维tensor的列表！
                    Llist.extend(subL)
                else:
                    Llist.append(subL)
                # Llist.append(subL) #
                # L+=(coeffis[losses_ind]*subL) #
            nLlist=len(Llist)
            if coeffis_lossesname_flag and (len(coeffis)<nLlist or len(losses_name)<nLlist): #只修改一次，避免重复
                print('重置系数coeffis(%d)和损失名称losses_num(%d)，它们的长度应等于所有损失输出的数量和(%d)，coeffis全部重置为1！' \
                    %(len(coeffis),len(losses_name),nLlist))
                losses_name=['loss%d'%(i+1) for i in range(nLlist)]
                coeffis=[1]*nLlist
                coeffis_lossesname_flag=False
            L=sum([coeffis[i]*l for i,l in enumerate(Llist)])
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            if batch_ind==0 or (batch_ind+1)%20==0 or batch_ind+1==all_batches_num:
                opt_parms=optimizer.param_groups if scheduler is None else scheduler.optimizer.param_groups
                print('epoch=[%d/%d], batch=[%d/%d], %s, %s'%(epoch+1,epochs,batch_ind+1,all_batches_num, \
                    'loss=%f'%L if nLlist==1 else ('loss(all)=%f, '%L)+ \
                    (', '.join(['%s=%f'%(losses_name[i],Llist[i]) for i in range(nLlist)])), \
                    ', '.join(['lr(%s)=%s'%(e.get('name',i+1),'{:g}'.format(e['lr'])) for i,e in enumerate(opt_parms)]) \
                    if len(list(opt_parms))>1 else 'lr=%s'%('{:g}'.format(opt_parms[0]['lr']))))
        if scheduler is not None:
            scheduler.step()
        if checkpoint is not None:
            checkpoint.save({'state':net.module.state_dict() if \
                    isinstance(net,nn.DataParallel) else net.state_dict(),'epoch':epoch,
                    'optimizer':optimizer.state_dict(),
                    'scheduler':None if scheduler is None else scheduler.state_dict()})
    print('Train OVER!')