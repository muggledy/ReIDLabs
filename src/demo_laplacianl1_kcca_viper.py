'''
This method(unsupervised dictionary learning with graph laplacian L1 regularization, see ./code/udlgl.py) 
use features in [1], you can download from [2](VIPeR_split.mat), then put in ../data/, test with XQDA we 
get rank-1 34.15%(10 iters),
see author's web page: https://sites.google.com/site/elyorkodirovresearch/publications
References:
[1] Giuseppe Lisanti et al., Matching People across Camera Views using Kernel Canonical Correlation 
    Analysis”, Eighth ACM/IEEE International Conference on Distributed Smart Cameras, 2014.
[2] https://github.com/glisanti/KCCAReId
'''

import numpy as np
import os.path
import scipy.io as sio
from code.lomo.xqda import xqda
from code.lomo.tools import mah_dist,calc_cmc,plot_cmc
from code.tools import norm_labels,cosine_dist,norm_labels_simultaneously
from code.optimize import opt_coding_l2
from code.udlgl import udlgl

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
cs=[]
for trial in range(0,10):
    data=get_data(trial)
    probFea1,galFea1=data['trainA_feats'],data['trainB_feats']
    probFea2,galFea2=data['testA_feats'],data['testB_feats']

    W,M,*_=xqda(probFea1,galFea1,*norm_labels_simultaneously(data['trainA_labels'][0],data['trainB_labels'][0]))
    dist=mah_dist(M,(W.T).dot(probFea2),(W.T).dot(galFea2))
    c=calc_cmc(dist.T,*norm_labels_simultaneously(data['testA_labels'][0],data['testB_labels'][0]),100)
    cs.append(c)

c_mean=np.mean(cs,axis=0)
print('CMC:')
print(c_mean)
plot_cmc(c_mean,['viper'],verbose=True)
###

# if __name__=='__main__':
#     cs=[]
#     lambd=1 #
#     for trial in range(0,10):
#         data=get_data(trial)
#         probFea1,galFea1=data['trainA_feats'],data['trainB_feats']
#         probFea2,galFea2=data['testA_feats'],data['testB_feats']

#         D=udlgl(probFea1,galFea1,k=5,lambd1=1,lambd2=1,gamma=0.1,nBasis=None,rho=1.4)
#         dist=cosine_dist(opt_coding_l2(D,probFea2,lambd),opt_coding_l2(D,galFea2,lambd))
#         c=calc_cmc(dist.T,norm_labels(data['testA_labels'][0]),norm_labels(data['testB_labels'][0]),100)
#         cs.append(c)
#         break
#     c_mean=np.mean(cs,axis=0)
#     print('CMC:')
#     print(c_mean)
#     plot_cmc(c_mean,['viper'],verbose=True)