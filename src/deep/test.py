import torch as pt
import torch.nn as nn
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../'))
from zoo.tools import measure_time,print_cmc,cosine_dist,euc_dist,print_if_visual
from deep.re_ranking import re_ranking
from deep.data_loader import testDataset
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as scio
from itertools import count
from functools import partial

def cosine_dist_T(X,Y,desc=('query','gallery'),if_print=True):
    if if_print:
        print('Calc cosine distance between %s%s and %s%s'%(desc[0],str(X.shape),desc[1],str(Y.shape)))
    return cosine_dist(X.T,Y.T)

def euc_dist_T(X,Y,desc=('query','gallery'),if_print=True):
    if if_print:
        print('Calc euclidean distance between %s%s and %s%s'%(desc[0],str(X.shape),desc[1],str(Y.shape)))
    return euc_dist(X.T,Y.T)

def extract_feats(net,data_iter,device=None,**kwargs):
    '''data_iter除了可以是一个DataLoader实例，还可以是一个文件夹路径，不过此时必须传递transform关键字参数'''
    if isinstance(data_iter,str) and os.path.isdir(data_iter):
        if kwargs.get('transform') is None:
            raise ValueError('Must give transform for func extract_feats!')
        data_iter=DataLoader(testDataset(data_iter,kwargs['transform']),batch_size= \
            kwargs.get('batch_size',32),shuffle=False,num_workers=kwargs.get('num_workers',4), \
            drop_last=False)

    device=pt.device('cuda' if pt.cuda.is_available() else 'cpu') if device is None else device
    net=net.to(device)
    net.eval()
    if isinstance(net,nn.DataParallel): #同test函数
        net.module.train_mode=False
    else:
        net.train_mode=False

    feats=[]
    with pt.no_grad():
        for batch,*_ in data_iter:
            batch=batch.to(device)
            feat=net(batch).data.cpu()
            feats.append(feat)
        feats=pt.cat(feats,0)
    feats=feats.numpy().T #(d,n)
    print('Extracted features%s successfully'%str(feats.shape))
    return feats

# @measure_time
def test(net,query_iter,gallery_iter,evaluate=None,ranks=[1,5,10,20,50,100],device=None,save_galFea=None, \
         calc_dist_funcs=None,alphas=None,re_rank=False,return_top_rank=False,if_print=True):
    '''如果gallery_iter不是DataLoader对象，而是普通路径字符串，表示gallery特征将从文件加载。
       如果save_galFea不是None而是路径字符串，则会保存经过网络提取的gallery特征至该路径'''
    device=pt.device('cuda' if pt.cuda.is_available() else 'cpu') if device is None else device
    printv=partial(print_if_visual,visual=if_print)

    net=net.to(device)
    net.eval()
    # net.train_mode=False
    if isinstance(net,nn.DataParallel): #我在train函数中调用了test函数，此时net可能是DataParallel对象
        net.module.train_mode=False
    else:
        net.train_mode=False

    if return_top_rank and evaluate is None:
        raise ValueError('Must give evaluate creation if want return top rank!')

    with pt.no_grad():
        q_feas,q_pids,q_cids=[],[],[]
        for i,(batchImgs,pids,cids) in enumerate(query_iter):
            batchImgs=batchImgs.to(device)
            batchFeas=net(batchImgs) #batchFeas may be list obj: [tensor,tensor,...]
            if isinstance(batchFeas,pt.Tensor):
                batchFeas=batchFeas.data.cpu()
            elif isinstance(batchFeas,(list,tuple)):
                batchFeas=[_.data.cpu() for _ in batchFeas]
            q_feas.append(batchFeas)
            q_pids.append(pids.data.cpu())
            q_cids.append(cids.data.cpu())
        if isinstance(q_feas[0],pt.Tensor):
            q_feas=pt.cat(q_feas,0).numpy()
        elif isinstance(q_feas[0],(list,tuple)):
            q_feas=[pt.cat(i,0).numpy() for i in zip(*q_feas)]
        printv("Extracted deep features%s for query set"%('[%s]'%(','.join([str(i.shape) for i in q_feas])) \
            if isinstance(q_feas,list) else str(q_feas.shape)))
        q_pids=pt.cat(q_pids).numpy()
        q_cids=pt.cat(q_cids).numpy()

        if isinstance(gallery_iter,str):
            gallery_iter=os.path.normpath(gallery_iter)
            data=scio.loadmat(gallery_iter)
            if data.get('g_feas_0') is not None:
                g_feas=[]
                for i in count():
                    if data.get('g_feas_%d'%i) is not None:
                        g_feas.append(data['g_feas_%d'%i])
                    else:
                        break
            else:
                g_feas=data['g_feas']
            g_pids=data['g_pids'].reshape(-1) #注意保存的(n,)一维数组读取后会变成(1,n)二维数组，可能导致错误
            g_cids=data['g_cids'].reshape(-1)
            printv('Loaded gallery features%s from %s'%('[%s]'%(','.join([str(i.shape) for i in g_feas])) \
                if isinstance(g_feas,list) else str(g_feas.shape),gallery_iter))
        else:
            g_feas,g_pids,g_cids=[],[],[]
            for i,(batchImgs,pids,cids) in enumerate(gallery_iter):
                batchImgs=batchImgs.to(device)
                batchFeas=net(batchImgs)
                if isinstance(batchFeas,pt.Tensor):
                    batchFeas=batchFeas.data.cpu()
                elif isinstance(batchFeas,(list,tuple)):
                    batchFeas=[_.data.cpu() for _ in batchFeas]
                g_feas.append(batchFeas)
                g_pids.append(pids.data.cpu())
                g_cids.append(cids.data.cpu())
            if isinstance(g_feas[0],pt.Tensor):
                g_feas=pt.cat(g_feas,0).numpy()
            elif isinstance(g_feas[0],(list,tuple)):
                g_feas=[pt.cat(i,0).numpy() for i in zip(*g_feas)]
            printv("Extracted deep features%s for gallery set"%('[%s]'%(','.join([str(i.shape) for i in g_feas])) \
                if isinstance(g_feas,list) else str(g_feas.shape)))
            g_pids=pt.cat(g_pids).numpy()
            g_cids=pt.cat(g_cids).numpy()
            if save_galFea is not None:
                save_galFea=os.path.normpath(save_galFea)
                printv('Save gallery features into %s'%save_galFea)
                if isinstance(g_feas,np.ndarray):
                    d={'g_feas':g_feas,'g_pids':g_pids,'g_cids':g_cids}
                elif isinstance(g_feas,list):
                    d=dict(zip(['g_feas_%d'%i for i in range(len(g_feas))],g_feas))
                    d['g_pids']=g_pids
                    d['g_cids']=g_cids
                scio.savemat(save_galFea,d)
                
    #话说这是什么距离？来自：https://github.com/michuanhaohao/deep-person-reid/blob/master/train_img_model_xent.py
    # m,n=q_feas.size(0),g_feas.size(0)
    # distmat=pt.pow(q_feas,2).sum(dim=1,keepdim=True).expand(m,n)+ \
    #         pt.pow(g_feas,2).sum(dim=1,keepdim=True).expand(n,m).t()
    # distmat.addmm_(1,-2,q_feas,g_feas.t())
    # distmat=distmat.numpy()
    if isinstance(q_feas,np.ndarray):
        q_feas=[q_feas]
    if isinstance(g_feas,np.ndarray):
        g_feas=[g_feas]
    if calc_dist_funcs is None: #需要注意的是，距离计算函数的输入对象形状必须是(sample_num,...)
        printv('Using default dist evaluation function: cosine%s'%('(all)' if len(q_feas)>1 else ''))
        calc_dist_funcs=[cosine_dist_T for i in range(len(q_feas))]
    else:
        if not isinstance(calc_dist_funcs,(list,tuple)):
            calc_dist_funcs=[calc_dist_funcs]
        if len(calc_dist_funcs)!=len(q_feas):
            raise ValueError('Invalid calc_dist_funcs, it\'s length must be %d'%len(q_feas))
    if alphas is None:
        if len(q_feas)>1:
            printv('Using default fusion coefficient: 1(all)')
        alphas=[1]*len(q_feas)
    else:
        if not isinstance(alphas,(list,tuple)):
            alphas=[alphas]
        if len(alphas)!=len(q_feas):
            raise ValueError('Invalid alphas, it\'s length must be %d'%len(q_feas))
    distmat=0
    all_dists=[]
    if len(q_feas)!=len(g_feas):
        printv('WARNING: the number of features type of query(%d) is different from gallery(%d), check it, it may cause error!' \
            %(len(q_feas),len(g_feas)))
    for i,(qf,gf) in enumerate(zip(q_feas,g_feas)):
        sub_dist=calc_dist_funcs[i](qf,gf,if_print=if_print)
        if re_rank:
            sub_dist=re_ranking(sub_dist,calc_dist_funcs[i](qf,qf,('query','query'),if_print=if_print), \
                calc_dist_funcs[i](gf,gf,('gallery','gallery'),if_print=if_print))
        distmat+=(alphas[i]*sub_dist)
        all_dists.append(sub_dist)

    if evaluate is not None:
        if len(all_dists)>1:
            for i,sub_dist in enumerate(all_dists):
                printv('Calc CMC and mAP with sub dist matrix %d:'%i)
                cmc,mAP=evaluate(sub_dist.T,q_pids,g_pids,q_cids,g_cids,max(ranks))
                print_cmc(cmc,color=False)
                printv('mAP:%.2f'%(mAP*100))
        printv("Computing CMC and mAP%s..."%('(fusion result)' if len(all_dists)>1 is not None else ''))
        cmc,mAP=evaluate(distmat.T,q_pids,g_pids,q_cids,g_cids,max(ranks))
        if return_top_rank:
            return cmc[0],mAP
        print_cmc(cmc,color=True)
        printv('mAP:%.2f'%(mAP*100))
    y=np.argsort(distmat)
    # x=broadcast_to(np.arange(distmat.shape[0])[:,None],distmat.shape)
    # match_id=np.broadcast_to(g_pids,distmat.shape)[x,y] #每一行代表一个query的匹配结果，元素为匹配行人的id，不过这个没啥用
    return y #see in ./plot_match.py
