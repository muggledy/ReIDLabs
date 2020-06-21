import torch as pt
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from tools import measure_time,print_cmc,cosine_dist

@measure_time
def test(net,query_iter,gallery_iter,evaluate,ranks=[1,5,10,20,50,100],device=None):
    flag='cuda' if pt.cuda.is_available() else 'cpu'
    if device is None:
        device=pt.device(flag)
    net=net.to(device)
    net.eval()
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
        print("Extracted deep features(%d,%d) from query set"%(q_feas.size(0),q_feas.size(1)))

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
        print("Extracted deep features(%d,%d) from gallery set"%(g_feas.size(0),g_feas.size(1)))

    #话说这是什么距离？来自：https://github.com/michuanhaohao/deep-person-reid/blob/master/train_img_model_xent.py
    # m,n=q_feas.size(0),g_feas.size(0)
    # distmat=pt.pow(q_feas,2).sum(dim=1,keepdim=True).expand(m,n)+ \
    #         pt.pow(g_feas,2).sum(dim=1,keepdim=True).expand(n,m).t()
    # distmat.addmm_(1,-2,q_feas,g_feas.t())
    # distmat=distmat.numpy()
    distmat=cosine_dist(q_feas.numpy().T,g_feas.numpy().T)

    print("Computing CMC and mAP")
    cmc,mAP=evaluate(distmat.T,q_pids,g_pids,q_cids,g_cids,max(ranks))
    print_cmc(cmc,color=True)
    print('mAP:%.2f'%(mAP*100))
    return cmc[0]