'''
References:
[1] Min Cao,Chen Chen,Xiyuan Hu,et al. Towards fast and kernelized orthogonal discriminant 
    analysis on person re-identification[J]. Pattern Recognition,2019(94):218-229.
muggledy 2020/4/5
'''

import numpy as np
from numpy.linalg import inv,svd,qr
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from zoo.lda import get_Hw_Hb_Ht
from zoo.klda import get_kernel_Hw_Hb_Ht
from lomo.tools import measure_time
from functools import reduce,partial

@measure_time
def olda(X,labels):
    '''X is the training set which X[:,i] represents ith sample and labels[i] represents 
       the ID of X[:,i]. Func will return the project matrix L of LDA, give a test 
       sample x, use Lᵀx'''
    _,Hb,Ht=get_Hw_Hb_Ht(X,labels)
    Ut,convt,_=svd(Ht,full_matrices=False)
    inv_convt=inv(np.diag(convt))
    T=inv_convt.dot(Ut.T).dot(Hb)
    P,*_=svd(T,full_matrices=False) #at first i don't set full_matrices to False, so 
                                    #the rank-1 is very low
    G=Ut.dot(inv_convt).dot(P)
    Q,_=qr(G)
    return Q

@measure_time
def kolda(X,labels,kernel='gaussian',**kwargs):
    '''kernel version of olda. Note the returns(A,func) of this function, for test samples 
       Y, use Aᵀfunc(Y), i.e. Aᵀk(X,Y), k is the kernel function'''
    _,Hb,Ht,func=get_kernel_Hw_Hb_Ht(X,labels,kernel,**kwargs)

    Ut,convt,_=svd(Ht,full_matrices=False)
    inv_convt=inv(np.diag(convt))
    T=inv_convt.dot(Ut.T).dot(Hb)
    P,*_=svd(T,full_matrices=False)
    G=Ut.dot(inv_convt).dot(P)
    A,_=qr(G)
    
    return A,func