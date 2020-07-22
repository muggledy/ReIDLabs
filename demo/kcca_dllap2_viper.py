'''
This method(unsupervised dictionary learning with graph laplacian L2 regularization) use features 
in [2], you can download from [3](VIPeR_split.mat), then put in ../data/, test with XQDA we 
get rank-1 34.15%(average after 10 iters)
Because of the complexity of it's optimization part(see paper 
"Graph Regularized Sparse Coding for Image Representation"), i.e. feature sign search, so i 
modify the objective function(||Y||_1=>||Y||_F) accroding to my teacher's guidance
Issue:
Paper[1] claims that rank-1 of viper can achieve 30%(cam_b as probe), but in fact there is a 
flaw in the public source code, only has 23% if you shuffle the training data (test in matlab, 
but i got 25% with python code), 
the problem lies in the construction of the affinity matrix W(see graph_cross_view.m in source code)
see author's web page: https://sites.google.com/site/elyorkodirovresearch/publications
2020/5/21
References:
[1] Kodirov, Elyor & Xiang, Tao & Gong, Shaogang. (2015). Dictionary Learning with Iterative 
    Laplacian Regularisation for Unsupervised Person Re-identification. 44.1-44.12. 10.5244/C.29.44.
[2] Giuseppe Lisanti et al., Matching People across Camera Views using Kernel Canonical Correlation 
    Analysis”, Eighth ACM/IEEE International Conference on Distributed Smart Cameras, 2014.
[3] https://github.com/glisanti/KCCAReId
'''

from initial import *
from lomo.xqda import xqda
from zoo.dllap import get_cross_view_graph,dllap
from zoo.optimize import opt_coding_l2
from zoo.tools import norm_labels_simultaneously
import scipy.io as sio

def get_data(trial=0,dataset='viper'):
    file_path=os.path.join(os.path.dirname(__file__),'../data/%s_split.mat'%dataset)
    data=sio.loadmat(file_path)['trials'][0]
    s='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    trial_num=len(data)

    if trial<0 or trial>=trial_num:
        raise ValueError('trial must be in [0,%d]!'%(trial_num-1))
    data_trial=data[trial]
    inds=list(zip(range(0,len(data_trial),2),range(1,len(data_trial),2)))
    test_inds=inds[:len(inds)//2]
    train_inds=inds[len(inds)//2:]
    ret={}
    for i,(labels,feats) in enumerate(train_inds):
        cam=s[i]
        ret['train%s_labels'%cam]=data_trial[labels]
        ret['train%s_feats'%cam]=data_trial[feats]
    for i,(labels,feats) in enumerate(test_inds):
        cam=s[i]
        ret['test%s_labels'%cam]=data_trial[labels]
        ret['test%s_feats'%cam]=data_trial[feats]
    return ret

### test of kcca descriptors + XQDA
# cs=[]
# for trial in range(0,10):
#     data=get_data(trial)
#     probFea1,galFea1=data['trainA_feats'],data['trainB_feats']
#     probFea2,galFea2=data['testA_feats'],data['testB_feats']

#     W,M,*_=xqda(probFea1,galFea1,*norm_labels_simultaneously(data['trainA_labels'][0],data['trainB_labels'][0]))
#     dist=mah_dist(M,(W.T).dot(probFea2),(W.T).dot(galFea2))
#     c=calc_cmc(dist.T,*norm_labels_simultaneously(data['testA_labels'][0],data['testB_labels'][0]),100)
#     cs.append(c)

# c_mean=np.mean(cs,axis=0)
# print('CMC:')
# print(c_mean)
# plot_cmc(c_mean,['viper'],verbose=True)
###

if __name__=='__main__':
    k_nn=3
    nBasis=2**8
    alpha=1.0
    beta=0.0001
    nIters=50
    lambd1=0.04

    cs=[]
    D=None
    for trial in range(1):
        print('Trial-%d...'%(trial+1))
        data=get_data(trial)
        probFea1,galFea1=data['trainA_feats'],data['trainB_feats']
        probFea1,galFea1=probFea1[:,np.random.permutation(probFea1.shape[1])], \
            galFea1[:,np.random.permutation(galFea1.shape[1])] #verify: shuffle samples' order won't 
                                                               #affect result if don't use backdoor 
                                                               #in func get_cross_view_graph, this result 
                                                               #is reliable
        probFea2,galFea2=data['testA_feats'],data['testB_feats']
        
        for in_iter in range(1):
            print('1.construct graph(W) similarity matrix...')
            if in_iter==0:
                W_full=get_cross_view_graph(probFea1,galFea1,k=k_nn,backdoor=False)
            elif in_iter==1:
                print('use the learned samples\' coding this time')
                W_full=get_cross_view_graph(train_a_after_D,train_b_after_D,k=k_nn,backdoor=False)

            print('2.learn dictionary(D) with L2 laplacian...')
            D=dllap(np.hstack((probFea1,galFea1)), W_full, nBasis, alpha, beta, nIters, Dinit=D) #
                                                        #非常奇怪的是，有时候收敛，有时候不收敛

            train_a_after_D,P=opt_coding_l2(D,probFea1,lambd1)
            train_b_after_D=P.dot(galFea1)

        test_a_after_D,P=opt_coding_l2(D,probFea2,lambd1)
        test_b_after_D=P.dot(galFea2)
        dist=cosine_dist(test_a_after_D,test_b_after_D)
        print('calc CMC...')
        c=calc_cmc(dist,*norm_labels_simultaneously(data['testB_labels'][0],data['testA_labels'][0]),100) #
                                                        #note here: cam_b as probe, if cam_a as probe, 
                                                        #rank scores will decrease a little
        cs.append(c)
        print_cmc(c)

    print('average CMC:')
    c_mean=np.mean(np.array(cs),axis=0)
    print_cmc(c_mean,color=True)
    plot_cmc(c_mean,['viper'],verbose=True)