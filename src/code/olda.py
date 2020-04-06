'''
References:
[1] Min Cao,Chen Chen,Xiyuan Hu,et al. Towards fast and kernelized orthogonal discriminant 
    analysis on person re-identification[J]. Pattern Recognition,2019(94):218-229.
muggledy 2020/4/5
'''

import numpy as np
from numpy.linalg import inv,svd,qr
from .lda import get_Hw_Hb_Ht
from .tools import norm_labels
from .lomo.tools import measure_time

@measure_time
def olda(X,labels):
    '''X is the training set which X[:,i] represents one sample and labels[i] represents 
       the ID of each sample. Func will return the project matrix L of LDA, give a test 
       sample x, use Lᵀx'''
    _,Hb,Ht=get_Hw_Hb_Ht(X,labels)
    Ut,convt,_=svd(Ht,full_matrices=False)
    inv_convt=inv(np.diag(convt))
    T=inv_convt.dot(Ut.T).dot(Hb)
    P,*_=svd(T)
    G=Ut.dot(inv_convt).dot(P)
    Q,_=qr(G)
    return Q

@measure_time
def fast_olda(X,labels):
    '''fast version of OLDA'''
    pass

@measure_time
def kolda():
    pass