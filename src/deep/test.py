import torch as pt
import torch.nn as nn
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from zoo.tools import measure_time,print_cmc,cosine_dist
import numpy as np

@measure_time
def test(net,query_iter,gallery_iter,evaluate=None,ranks=[1,5,10,20,50,100],device=None,save_galFea=None):
    '''如果gallery_iter不是DataLoader对象，而是普通路径字符串，表示gallery特征将从文件加载。
       如果save_galFea不是None而是路径字符串，则会保存经过网络提取的gallery特征至该路径'''
    device=pt.device('cuda' if pt.cuda.is_available() else 'cpu') if device is None else device
    net=net.to(device)
    net.eval()
    if isinstance(net,nn.DataParallel):
        net.module.train_mode=False
    else:
        net.train_mode=False
    with pt.no_grad():
        q_feas,q_pids,q_cids=[],[],[]
        for i,(batchImgs,pids,cids) in enumerate(query_iter):
            batchImgs=batchImgs.to(device)
            batchFeas=net(batchImgs).data.cpu()
            q_feas.append(batchFeas)
            q_pids.append(pids.data.cpu())
            q_cids.append(cids.data.cpu())
        q_feas=pt.cat(q_feas,0)
        q_pids=pt.cat(q_pids).numpy()
        q_cids=pt.cat(q_cids).numpy()
        print("Extracted deep features(%d,%d) for query set"%(q_feas.size(0),q_feas.size(1)))

        if isinstance(gallery_iter,str):
            gallery_iter=os.path.normpath(gallery_iter)
            print('Load gallery features from %s'%gallery_iter)
            data=np.load(gallery_iter)
            g_feas=pt.from_numpy(data['g_feas'])
            g_pids=data['g_pids']
            g_cids=data['g_cids']
        else:
            g_feas,g_pids,g_cids=[],[],[]
            for i,(batchImgs,pids,cids) in enumerate(gallery_iter):
                batchImgs=batchImgs.to(device)
                batchFeas=net(batchImgs).data.cpu()
                g_feas.append(batchFeas)
                g_pids.append(pids.data.cpu())
                g_cids.append(cids.data.cpu())
            g_feas=pt.cat(g_feas,0)
            g_pids=pt.cat(g_pids).numpy()
            g_cids=pt.cat(g_cids).numpy()
            print("Extracted deep features(%d,%d) for gallery set"%(g_feas.size(0),g_feas.size(1)))
            if save_galFea is not None:
                save_galFea=os.path.normpath(save_galFea)
                print('Save gallery features into %s'%save_galFea)
                np.savez(save_galFea,g_feas=g_feas.numpy(),g_pids=g_pids,g_cids=g_cids)

    #话说这是什么距离？来自：https://github.com/michuanhaohao/deep-person-reid/blob/master/train_img_model_xent.py
    # m,n=q_feas.size(0),g_feas.size(0)
    # distmat=pt.pow(q_feas,2).sum(dim=1,keepdim=True).expand(m,n)+ \
    #         pt.pow(g_feas,2).sum(dim=1,keepdim=True).expand(n,m).t()
    # distmat.addmm_(1,-2,q_feas,g_feas.t())
    # distmat=distmat.numpy()
    distmat=cosine_dist(q_feas.numpy().T,g_feas.numpy().T)
    if evaluate is not None:
        print("Computing CMC and mAP...")
        cmc,mAP=evaluate(distmat.T,q_pids,g_pids,q_cids,g_cids,max(ranks))
        print_cmc(cmc,color=True)
        print('mAP:%.2f'%(mAP*100))
    y=np.argsort(distmat)
    # x=broadcast_to(np.arange(distmat.shape[0])[:,None],distmat.shape)
    # match_id=np.broadcast_to(g_pids,distmat.shape)[x,y] #每一行代表一个query的匹配结果，元素为匹配行人的id，不过这个没啥用
    return y #see in ./plot_match.py