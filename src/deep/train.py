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
from visdom import Visdom
from deep.test import test
from tensorboardX import SummaryWriter
# from functools import partial

def setup_seed(seed):
    print('Use random seed(%d)'%seed)
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic=True

#可视化：https://zhuanlan.zhihu.com/p/220403674

def get_batch(data_iter, data_loader): #data_iter是enumerate(dataloader)对象，来自
                                       #https://github.com/yujheli/ARN/blob/master/reid_main.py
    try:
        _, batch = next(data_iter)
    except:
        data_iter = enumerate(data_loader)
        _, batch = next(data_iter)
    return batch, data_iter

@measure_time
def train(net,train_loader,losses,optimizer,epochs,scheduler=None,coeffis=None,device=None,checkpoint=None,use_amp=False, \
          amp_level='O1',losses_name=None,out_loss_map=None,if_visdom=False,if_tensorboard=False,tensorboard_subdir=None,
          use_pcids=None,**kwargs):
    '''注意，对于losses参数，即使只有一个损失，也要写为(loss,)或者[loss]，即序列形式。总体损失loss
       =coeffis[0]*losses[0](net_out[0],targets)+coeffis[1]*losses[1](net_out[1],targets)+...
       device的值可以是：str('cpu' or 'cuda'),scalar(0, index of GPU),list([0,1], indexes list 
       of GPU),None(优先使用GPU),torch.device,对于list，表示多卡训练，特别的，可以取值'DP'，表示
       使用全部GPU设备并行。通常损失仅仅使用了pids信息，而且是作为损失函数定义中的最后一个参数，如果
       要使用cids，或者两者都使用，再或者两者都不使用，必须显式提供use_pcids参数，取值为'P','C',
       'PC','CP'，譬如'P'表示仅使用pids，是默认值，而'PC'和'CP'的区别是pids在前还是cids在前，如果
       非这四种取值，则表示不使用任何监督信号，相应的损失定义也不含有pids和cids参数，use_pcids长度和
       losses参数一致。为了扩展到UDA场景，批训练数据train_loader包括两个来源，源域有标签数据和目标
       域无标签数据，此时定义为train_loader:=(source_train_loader,target_train_loader)，因此，在
       定义网络模型的forward输入时，也必须是source在前，target在后，且注意目标
       域没有行人标签，但具有摄像头标签，至于损失使用哪些监督信号，还是依靠use_pcids参数指定，其取值
       还是基本的四种，只不过在前面还有一个字符标识'S'或'T'，譬如'SP'表示使用源域行人标签，但是像'TP'
       显然是不合法的，因为目标域没有行人标签，在UDA情形下，网络的输入变成了两个训练批次，一是源域训
       练批次，二是目标域训练批次，另外我们设定源域在前，目标域在后'''
    UDA=False
    if isinstance(train_loader,(list,tuple)) and len(train_loader)==2:
        UDA=True
        print('Now is in Unsupervised Domain Adaption(UDA) scenario!' \
            '\nNote: train_loader:=(source_train_loader,target_train_loader) and net(source_train_imgs,target_train_imgs)')
        source_train_loader,target_train_loader=train_loader
        if len(iter(source_train_loader).__next__())!=3 or len(iter(target_train_loader).__next__())!=2: #对源域批训练数据和目标域批训练数据进行检查，
                                       #源域每一批batch应该包含图像、行人ID以及摄像头ID三项，而目标域是两项
            raise ValueError('Batch in source_train_loader must have imgs,' \
                'pids and cids, and batch in target_train_loader must have imgs and cids, please check it!')
    
    if if_visdom: #https://github.com/facebookresearch/visdom
        print('Startup visdom, please execute "python -m visdom.server"!')
        vis=Visdom(env='train') #python -m visdom.server
    if if_tensorboard:
        log_dir=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../data/tensorboard_logdir/', \
            '' if tensorboard_subdir is None else tensorboard_subdir))
        print('Startup tensorboard, please execute "python -m tensorboard.main --logdir=%s"!'%log_dir)
        # print('' if net.train_mode==True else 'WARN: To plot model in tensorboard, net.train_mode should be True!\n',end='')
        net.train();net.train_mode=True # ↑
        with SummaryWriter(log_dir=log_dir) as writer:
            if UDA:
                writer.add_graph(net,(iter(source_train_loader).__next__()[0],iter(target_train_loader).__next__()[0]))
            else:
                writer.add_graph(net,iter(train_loader).__next__()[0]) #需确保模型train_mode属性为True，否则在测试模式下，模型可能缺少一些组件，譬如分类层
    #运行：cd到event事件保存路径下，python -m tensorboard.main --logdir=./ 或者python -m tensorboard.main --logdir=具体的路径
    #切记不要给路径添加引号，否则显示空。https://blog.csdn.net/qq_23142123/article/details/80519535
    #https://blog.csdn.net/Aa545620073/article/details/89374112

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
        net, optimizer = amp.initialize(net, optimizer, opt_level=amp_level)
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

    if device.type=='cuda':
        cudnn.benchmark=True
        print('Set cudnn.benchmark %s'%(cudnn.benchmark))
        if DP_flag: #注意DataParallel包裹放在一切就绪之后，即模型net已经放到GPU设备上、加载好参数、设置好APEX等等之后
            gpu_num=pt.cuda.device_count()
            print('Use Data Parallel to Wrap Model',end='')
            net=nn.DataParallel(net,device_ids=device_ids) #允许多卡训练，但我收到警告：UserWarning: Pytorch is not compiled with NCCL support
                                 #参见https://github.com/pytorch/pytorch/issues/12277
                                 #https://discuss.pytorch.org/t/pytorch-windows-is-parallelization-over-multiple-gpus-now-possible/59971
                                 #https://www.programmersought.com/article/7854938878/，在工作站上的测试，双2080ti，结果相比单卡有所降低，时耗也增大一些
                                 #关于负载均衡的问题，参考https://www.cnblogs.com/zf-blog/p/12010742.html，但是该方法不适用Windows环境，
                                 #此外还可以用DistributedDataParallel解决
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
    
    if UDA:
        all_batches_num=max([len(source_train_loader),len(target_train_loader)])
    else:
        all_batches_num=len(train_loader)
    losses_num=len(losses)
    give_loss_name=True
    if losses_name is None: #实际losses_name的长度应等于所有损失函数的输出数量的总和，这边只是先赋一个初值，后面可能
                            #还会进行修改，因为有些损失函数的输出不止一个，譬如
                            #AlignedReID的三元组损失有两个输出，这种情况下建议人为给出losses_name，主要是使调用者清
                            #楚，打印时出现的子损失个数大于losses参数长度时亦不要吃惊
        losses_name=['loss%d'%(i+1) for i in range(losses_num)]
        give_loss_name=False
    if coeffis is None: #coeffis长度默认和losses_name保持一致，默认值全为1
        coeffis=[1]*len(losses_name)
    if use_pcids is None:
        if UDA:
            print('WARN: all losses only use source domain pids(SP) as supervision defaultly!')
            use_pcids=['SP' for i in range(losses_num)]
        else:
            print('WARN: all losses only use pids(P) as supervision defaultly!')
            use_pcids=['P' for i in range(losses_num)]
    if len(use_pcids)!=losses_num:
        raise ValueError('use_pcids\'s length must be the same as losses!')

    coeffis_lossesname_flag=True
    print_lossesinfo_flag=True
    
    for epoch in range(start_epoch,epochs):
        pt.cuda.synchronize()
        last_time=time.time()

        if UDA:
            source_train_iter=enumerate(source_train_loader)
            target_train_iter=enumerate(target_train_loader)
        else:
            train_iter=enumerate(train_loader)
        for batch_ind in range(all_batches_num):
            if UDA:
                (source_batchImgs,source_batchPids,source_batchCids),source_train_iter= \
                    get_batch(source_train_iter,source_train_loader)
                source_batchImgs,source_batchPids,source_batchCids=source_batchImgs.to(device), \
                    source_batchPids.to(device).to(pt.int64),source_batchCids.to(device).to(pt.int64)
                (target_batchImgs,target_batchCids),target_train_iter= \
                    get_batch(target_train_iter,target_train_loader)
                target_batchImgs,target_batchCids=target_batchImgs.to(device), \
                    target_batchCids.to(device).to(pt.int64)
            else:
                (batchImgs,batchPids,batchCids),train_iter=get_batch(train_iter,train_loader)
                batchImgs,batchPids,batchCids=batchImgs.to(device), \
                    batchPids.to(device).to(pt.int64),batchCids.to(device).to(pt.int64)
            if UDA:
                out=net(source_batchImgs,target_batchImgs)
            else:
                out=net(batchImgs)

            if not isinstance(out,(list,tuple)):
                out=[out]
            if out_loss_map is None and len(out)!=losses_num: #假设网络有n个输出，那么必须有相应的n个损失（这个n为参数losses长度），
                                                              #如果不能一一对应，则使用“输出-损失”映射out_loss_map来进行控制
                raise ValueError('The num(%d) of net\'s out must be equal to losses\' length(%d)!'%(len(out),losses_num))
            Llist=[]
            
            #除了指定的几种模式['P','C','PC','CP','SP','SC','SPC','SCP','TC']，以控制返回行人ID或摄像头ID，否则一律返回[]空，表示不使用任何监督信号
            if UDA:
                select_pcids={'SP':[source_batchPids],'SC':[source_batchCids],'SPC':[source_batchPids, \
                    source_batchCids],'SCP':[source_batchCids,source_batchPids],'TC':[target_batchCids]}
            else:
                select_pcids={'P':[batchPids],'C':[batchCids],'PC':[batchPids,batchCids],'CP':[batchCids,batchPids]}
            if print_lossesinfo_flag:
                print_losses_info=[]

            if out_loss_map is None:
                for losses_ind,loss in enumerate(losses):
                    if print_lossesinfo_flag: #只记录并打印一次
                        loss_record_info=[loss.__name__ if loss.__class__.__name__=='function' else loss.__class__.__name__, \
                            losses_ind,use_pcids[losses_ind] if use_pcids[losses_ind] \
                            in select_pcids.keys() else 'None',] #损失函数名称，损失输入（网络输出列表的下标），监督信号，
                                                                 #损失函数的输出（可能包含多个子损失）个数，输出（子损失）名称，输出（子损失）权重系数
                    subL=loss(out[losses_ind],*select_pcids.get(use_pcids[losses_ind],[]))
                    if isinstance(subL,(list,tuple)): #要求损失函数的输出要么是单个0维tensor（损失是标量），要么是多个0维tensor的列表！
                        Llist.extend(subL)
                        if print_lossesinfo_flag:
                            loss_record_info.append(len(subL))
                    else:
                        Llist.append(subL)
                        if print_lossesinfo_flag:
                            loss_record_info.append(1)
                    if print_lossesinfo_flag:
                        print_losses_info.append(loss_record_info)
            else: #关于out_loss_map参数的使用，譬如：[[(0,1),(0,)],[(2,3),(1,2)]]，这表示网络输出out的第一个和第二个值（张量）作为第一个损失的输入，而out
                  #的第三个输出和第四个输出分别是作为第二个损失和第三个损失的输入，所以一共有三个损失，那么coeffis损失系数列表应该包含三个权重值，一般情况下
                  #都是这样，但是还有一种特殊情况之前也已经介绍过，就是像AlignedReID那种一个损失函数输出多个子损失的，此处姑且假设第一个损失函数的输出包含两
                  #个值，那么拢共就应该有四个损失，相应的coeffis长度变为4。out_loss_map参数是可选的，默认为None，表示一一对应关系，即网络输出的第一个张量作
                  #为第一个损失的输入，第二个输出张量作为第二个损失的输入，依此类推，写法就是：[[(0,),(0,)],[(1,),(1,)]]
                for out_inds,loss_inds in out_loss_map:
                    if print_lossesinfo_flag:
                        losses_records_info=[[losses[loss_ind].__name__ if \
                            losses[loss_ind].__class__.__name__=='function' else \
                            losses[loss_ind].__class__.__name__, \
                            ','.join([str(_oiv) for _oiv in out_inds]), \
                            use_pcids[loss_ind] if use_pcids[loss_ind] in select_pcids.keys() \
                            else 'None',] for loss_ind in loss_inds]
                    subLs=[losses[loss_ind](*[out[out_ind] for out_ind in out_inds], \
                        *select_pcids.get(use_pcids[loss_ind],[])) for loss_ind in loss_inds]
                    for subL_ind,subL in enumerate(subLs):
                        if isinstance(subL,(list,tuple)):
                            Llist.extend(subL)
                            if print_lossesinfo_flag:
                                losses_records_info[subL_ind].append(len(subL))
                        else:
                            Llist.append(subL)
                            if print_lossesinfo_flag:
                                losses_records_info[subL_ind].append(1)
                    if print_lossesinfo_flag:
                        print_losses_info.extend(losses_records_info)
            nLlist=len(Llist)
            if coeffis_lossesname_flag and (len(coeffis)<nLlist or len(losses_name)<nLlist): #只修改一次，避免重复
                print('重置系数coeffis(%d)和损失名称losses_num(%d)，它们的长度应等于所有损失输出的数量和(%d)，coeffis全部重置为1！' \
                    %(len(coeffis),len(losses_name),nLlist))
                losses_name=['loss%d'%(i+1) for i in range(nLlist)]
                coeffis=[1]*nLlist
                coeffis_lossesname_flag=False

            if print_lossesinfo_flag:
                print('-'*127) #https://blog.csdn.net/weixin_42280517/article/details/80814677
                print('{1:{0}<10}|{2:{0}^12}|{3:{0}^6}|{4:{0}^6}|{5:{0}^15}|{6:{0}>12}'.format(chr(12288), \
                    '损失函数','输入（网络输出下标）','监督信号','输出个数','输出（包含子损失，自定义）','权重系数（对应子损失）'))
                print('-'*127)
                sub_loss_num=np.array([loss_record_info[3] for loss_record_info in print_losses_info])
                sub_loss_end=np.cumsum(sub_loss_num)
                sub_loss_start=np.concatenate((np.array([0]),sub_loss_end[:-1]))
                [loss_record_info.extend([','.join([str(_sln) for _sln in losses_name[sub_loss_start[loss_record_info_ind]: \
                    sub_loss_end[loss_record_info_ind]]]),','.join([str(_slc) for _slc in coeffis[sub_loss_start[loss_record_info_ind]: \
                    sub_loss_end[loss_record_info_ind]]])]) for loss_record_info_ind,loss_record_info \
                    in enumerate(print_losses_info)]
                for loss_record_info in print_losses_info:
                    print('{:<20}|{:^24}|{:^12}|{:^12}|{:^30}|{:>24}'.format(*[str(item) for item in loss_record_info]))
                print_lossesinfo_flag=False
                print('-'*127)

            L=sum([coeffis[i]*l for i,l in enumerate(Llist) if coeffis[i]!=0]) #注意此处，我添加了判断coeffis[i]!=0，如果系数为0，相应的子网络
                                                                               #不会进行反向传播，相比较于不加此判断时，即使子网络损失计算为0仍backward
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
                    (', '.join(['%s=%f%s'%(losses_name[i],Llist[i],'(*%s)'%str(coeffis[i]) if coeffis[i]!=1 else '') \
                    for i in range(nLlist)])), \
                    ', '.join(['lr(%s)=%s'%(e.get('name',i+1),'{:g}'.format(e['lr'])) for i,e in enumerate(opt_parms)]) \
                    if len(list(opt_parms))>1 else 'lr=%s'%('{:g}'.format(opt_parms[0]['lr'])),time_consume))
                
                vt_x=epoch*all_batches_num+batch_ind+1
                vt_y=[vt_l.detach().cpu().numpy().item()*vt_a for vt_l,vt_a in list(zip([L]+Llist,[1]+list(coeffis)))]
                vt_yn=['all']+['(sub)%s'%vt_ln for vt_ln in losses_name]
                if if_visdom: #https://blog.csdn.net/u011715038/article/details/106179313/
                    vis.line(X=[vt_x],Y=[vt_y],win='batch_win',update='append', \
                        opts=dict(title='batch loss',xlabel='batch',legend=vt_yn))
                if if_tensorboard:
                    with SummaryWriter(log_dir=log_dir) as writer:
                        writer.add_scalars('train/batch_loss',{vt_yni:vt_yi for vt_yi,vt_yni in list(zip(vt_y,vt_yn))},vt_x)
        if scheduler is not None:
            scheduler.step()
        if checkpoint is not None:
            checkpoint.save({**{'state':net.module.state_dict() if \
                    isinstance(net,nn.DataParallel) else net.state_dict(),'epoch':epoch,
                    'optimizer':optimizer.state_dict(),
                    'scheduler':None if scheduler is None else scheduler.state_dict()},**({'amp':amp.state_dict()} if use_amp else {})})
        if kwargs.get('query_iter') and kwargs.get('gallery_iter') and kwargs.get('evaluate'):
            print('Calc CMC of query and gallery(TestSet)',end='')
            if UDA:
                if kwargs.get('uda_test_who') is None:
                    raise ValueError('Must tell which(source or target) you want to test explicitly in UDA!')
                else:
                    print(' for %s domain in UDA'%kwargs['uda_test_who'],end='')
            # else:
            #     if kwargs.get('uda_test_who') is not None:
            #         print('WARN: param uda_test_who only used for UDA! deleting it')
            #         del kwargs['uda_test_who']
            print(', ',end='')
            rank1,mAP=test(net,device=device,return_top_rank=True,if_print=False,**kwargs) #kwargs参数全部传递给test函数，除了query_iter、gallery_iter、evaluate
                                                                            #是必须传递之外，还有calc_dist_funcs、alphas、re_rank这些则是可选的
            net.train() #由于test函数中更改网络为eval模式，此处重置回train模式，然后继续训练
            if isinstance(net,nn.DataParallel):
                net.module.train_mode=True
            else:
                net.train_mode=True
            if if_visdom or if_tensorboard:
                print('see result in localhost(%s)'%('visdom,tensorboard' if if_visdom and if_tensorboard else ('visdom' if if_visdom else 'tensorboard')))
                if if_visdom:
                    vis.line(X=np.array([[epoch+1,epoch+1]]),Y=np.array([[rank1*100,mAP*100]]), \
                        win='rank1_mAP_win',update='append',opts=dict(legend=['rank-1','mAP'],title='rank-1 and mAP',markers='.',xlabel='epoch'))
                if if_tensorboard:
                    with SummaryWriter(log_dir=log_dir) as writer:
                        writer.add_scalars('train/rank1_mAP',{'rank-1':rank1*100,'mAP':mAP*100},epoch+1)
            else:
                print('rank-1:%.2f%% and mAP:%f'%(rank1*100,mAP*100))
    print('Train OVER!')