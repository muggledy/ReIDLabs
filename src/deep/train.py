import torch as pt
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../'))
from zoo.tools import measure_time
import time
# from functools import partial

def setup_seed(seed):
    print('Use random seed(%d)'%seed)
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic=True

@measure_time
def train(net,train_iter,losses,optimizer,epochs,scheduler=None,coeffis=None,device=None,checkpoint=None,use_amp=False,**kwargs):
    '''注意，对于losses参数，即使只有一个损失，也要写为(loss,)或者[loss]，即序列形式。总体损失loss
       =coeffis[0]*losses[0](net_out[0],targets)+coeffis[1]*losses[1](net_out[1],targets)+...
       device的值可以是：str('cpu' or 'cuda'),scalar(0, index of GPU),list([0,1], indexes list 
       of GPU),None(优先使用GPU),torch.device,对于list，表示多卡训练，特别的，可以取值'DP'，表示
       使用全部GPU设备并行'''
    if use_amp:
        print('Use AMP(Automatic Mixed Precision)')
        from apex import amp
    DP_flag=False #Data Parallel Flag
    if device is None:
        device=pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu') #此时如果存在GPU，优先使用GPU。注意
                                    #当使用'cuda'作为运行设备时，与使用'cuda:X'（torch.cuda.current_device()）是一致的
                                    #当前使用的设备总是返回0（号逻辑设备）
                                    #https://discuss.pytorch.org/t/torch-cuda-current-device-always-return-0/26530
    elif isinstance(device,str):
        if device in ['cpu','cuda']:
            device=pt.device(device)
        elif device=='DP': #此时DataParallel会自动检测并使用所有可用的GPU设备，(逻辑)设备列表记作g（0,1,2,...）,总是一个从0增长的序列
                           #但是具体映射到哪个物理设备，是由初始设置的os.environ['CUDA_VISIBLE_DEVICES']最终决定
            DP_flag=True
            device_ids=None
            device=pt.device('cuda',0) #所以这里应该设为g[0]，也就是0号逻辑设备，参见https://www.cnblogs.com/marsggbo/p/10962763.html
                                       #If you have 4 gpus: 0, 1, 2, 3. And run CUDA_VISIBLE_DEVICES=1,2 in xxx.py. 
                                       #Then the device that you will see within python are device 0, 1. Using 
                                       #device 0 in your code will use device 1 from global numering. Using device 1 
                                       #in your code will use 2 outside.
    elif isinstance(device,int):
        device=pt.device('cuda',device)
    elif isinstance(device,list):
        DP_flag=True
        device_ids=device
        device=pt.device('cuda',device[0]) #pt.device总是指定的是逻辑设备。当采用DP时，逻辑设备列表的第一个才是“主设备”，模型、数据都是拷贝到此设备上
    elif isinstance(device,pt.device):
        pass
    else:
        raise ValueError('Invalid device!')

    net=net.to(device)
    if use_amp: #参考：https://blog.csdn.net/qq_34914551/article/details/103203862
                #https://blog.csdn.net/ccbrid/article/details/103207676
                #https://nvidia.github.io/apex/amp.html
        net, optimizer = amp.initialize(net, optimizer, opt_level=kwargs.get('amp_level',"O1"))
        #训练伊始，有可能会出现：Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 32768.0
        #https://github.com/NVIDIA/apex/issues/635
        #https://blog.csdn.net/zjc910997316/article/details/103559837
        #https://zhuanlan.zhihu.com/p/79887894
    
    if checkpoint is not None and checkpoint.loaded:
        net.load_state_dict(checkpoint.states_info_etc['state']) #注意从本地加载的模型参数其device属性是cuda，所以在此之前net必须先to(cuda)
        start_epoch=checkpoint.states_info_etc['epoch']+1
        optimizer.load_state_dict(checkpoint.states_info_etc['optimizer']) #必须配合保存重载optimizer，scheduler才起作用
        for state in optimizer.state.values(): #https://blog.csdn.net/weixin_41848012/article/details/105675735
            for k, v in state.items():
                if pt.is_tensor(v):
                    state[k] = v.to(device)
        if checkpoint.states_info_etc.get('scheduler'):
            scheduler.load_state_dict(checkpoint.states_info_etc['scheduler'])
        if use_amp: #注意是在amp.initialize之后加载本地参数
            amp.load_state_dict(checkpoint.states_info_etc['amp'])

        checkpoint.states_info_etc=None #在训练MGN模型时，采用PK采样（用于三元组损失），P=8，K=4，事实上此时显存占用已达60%-70%（8G），
        #当我暂停训练，并从checkpoint继续时，得到一条错误：CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
        #隐约感到这是显存不够导致的，我降低了批次大小，令P=2，不再报错，于是定位到此处，在从checkpoint加载模型参数后，将checkpoint置为None
        #（以释放空间）
        checkpoint.loaded=False #与此同时，将loaded标志置为False，下次必须重新加载才行
    else:
        start_epoch=0
    # if scheduler is not None and isinstance(scheduler,partial):
    #     scheduler=scheduler(last_epoch=start_epoch-1)
    # if scheduler is not None:
    #     scheduler.step(start_epoch-1)

    if device.type=='cuda':
        cudnn.benchmark=True
        print('Set cudnn.benchmark %s'%(cudnn.benchmark))
        if DP_flag: #注意DataParallel包裹放在一切就绪之后，即模型net已经放到GPU设备上、加载好参数、设置好APEX等等之后
            gpu_num=pt.cuda.device_count()
            print('Use Data Parallel to Wrap Model',end='')
            net=nn.DataParallel(net,device_ids=device_ids) #允许多卡训练，https://blog.csdn.net/zhjm07054115/article/details/104799661/
                                 #似乎Windows并不支持多卡训练，我收到警告：UserWarning: Pytorch is not compiled with NCCL support
                                 #https://github.com/pytorch/pytorch/issues/12277
                                 #https://discuss.pytorch.org/t/pytorch-windows-is-parallelization-over-multiple-gpus-now-possible/59971
                                 #https://www.programmersought.com/article/7854938878/
                                 #在工作站上的测试，双2080ti，但不仅是rank-1还是mAP都比我的2070SUPER单卡降低了2个点，而且时间
                                 #还有所增加。好吧，直接忽略该警告即可
            if gpu_num==1:
                print(', but you have only one GPU')
            else:
                print('\nTraining on %d GPU devices...'%gpu_num)
        else:
            print('Running on device %s...'%device)
    else:
        print('Now training on %s(But GPU is proposed to accelerate)...'%device)

    net.train() #这一段也可以放到DataParallel包裹之前，简化为：net.train();net.train_mode=True
    if isinstance(net,nn.DataParallel):
        net.module.train_mode=True
    else:
        net.train_mode=True
    
    all_batches_num=len(train_iter)
    losses_num=len(losses)
    losses_name=kwargs.get('losses_name')
    give_loss_name=True
    if losses_name is None: #实际losses_name的长度应等于所有损失函数的输出数量的总和，这边只是先赋一个初值，后面可能
                            #还会进行修改，因为有些损失函数的输出不止一个，譬如
                            #AlignedReID的三元组损失有两个输出，这种情况下建议人为给出losses_name，主要是使调用者清
                            #楚，打印时出现的子损失个数大于losses参数长度时不要吃惊
        losses_name=['loss%d'%(i+1) for i in range(losses_num)]
        give_loss_name=False
    if coeffis is None: #coeffis长度默认和losses_name保持一致，默认值全为1
        coeffis=[1]*len(losses_name)
    coeffis_lossesname_flag=True
    
    for epoch in range(start_epoch,epochs):
        pt.cuda.synchronize()
        last_time=time.time()
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
            if use_amp:
                with amp.scale_loss(L, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                L.backward()
            optimizer.step()
            if batch_ind==0 or (batch_ind+1)%20==0 or batch_ind+1==all_batches_num:
                pt.cuda.synchronize()
                new_last_time=time.time()
                time_consume=new_last_time-last_time
                last_time=new_last_time
                opt_parms=optimizer.param_groups if scheduler is None else scheduler.optimizer.param_groups
                print('epoch=[%d/%d], batch=[%d/%d], %s, %s, time=%.2fs'%(epoch+1,epochs,batch_ind+1,all_batches_num, \
                    ('loss=%f'%L if not give_loss_name else '%s=%f'%(losses_name[0],L)) if nLlist==1 else ('loss(all)=%f, '%L)+ \
                    (', '.join(['%s=%f'%(losses_name[i],Llist[i]) for i in range(nLlist)])), \
                    ', '.join(['lr(%s)=%s'%(e.get('name',i+1),'{:g}'.format(e['lr'])) for i,e in enumerate(opt_parms)]) \
                    if len(list(opt_parms))>1 else 'lr=%s'%('{:g}'.format(opt_parms[0]['lr'])),time_consume))
        if scheduler is not None:
            scheduler.step()
        if checkpoint is not None:
            checkpoint.save({**{'state':net.module.state_dict() if \
                    isinstance(net,nn.DataParallel) else net.state_dict(),'epoch':epoch,
                    'optimizer':optimizer.state_dict(),
                    'scheduler':None if scheduler is None else scheduler.state_dict()},**({'amp':amp.state_dict()} if use_amp else {})})
    print('Train OVER!')