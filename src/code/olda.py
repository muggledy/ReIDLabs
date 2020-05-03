'''
References:
[1] Min Cao,Chen Chen,Xiyuan Hu,et al. Towards fast and kernelized orthogonal discriminant 
    analysis on person re-identification[J]. Pattern Recognition,2019(94):218-229.
muggledy 2020/4/5
'''

import numpy as np
from numpy.linalg import inv,svd,qr
from .lda import get_Hw_Hb_Ht
from .tools import norm_labels,euc_dist
from .lomo.tools import measure_time
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

def kernel_gaussian(X,Y=None,**kwargs):
    '''calc kernel matric of X(d,n1) and Y(d,n2) with gaussian kernel. Y will be the same 
       as X if Y==None defaulty'''
    sigma=kwargs.get('sigma',1) #in fact, this super parameter may be different with 
                                #different dataset, sigma is set as 1 for viper, 2 for
                                #prid2011, 0.5 for cuhk01 and cuhk03 in paper[1]
    Y=X if Y is None else Y
    D=euc_dist(X,Y)
    D=(-D)/(2*sigma**2)
    return np.exp(D)

def kernel_linear(X,Y=None):
    '''linear kernel function'''
    Y=X if Y is None else Y
    return X.T.dot(Y)

@measure_time
def kolda(X,labels,kernel='gaussian',**kwargs):
    '''kernel version of olda. Note the returns(A,func) of this function, for test samples 
       Y, use Aᵀfunc(Y), equal to AᵀK(X,Y), K is the kernel function'''
    labels=norm_labels(labels)
    sortedInd=np.argsort(labels)
    labels=labels[sortedInd]
    X=X[:,sortedInd]
    groups=reduce \
        (lambda x,y:x+[[y]] if x[-1][-1]!=y else x[:-1]+[x[-1]+[y]],labels,[['^']])[1:]

    print('use %s kernel'%kernel)
    if kernel=='gaussian': #RBF
        K=kernel_gaussian(X,None,**kwargs)
        func=partial(kernel_gaussian,X,**kwargs)
    elif kernel=='sigmoid':
        pass
    elif kernel=='linear':
        K=kernel_linear(X,None)
        func=partial(kernel_linear,X)
    else:
        raise ValueError('invalid kernel function!')
    
    _,n=X.shape
    B=np.zeros((n,n))
    start=0
    for g in groups:
        l=len(g)
        B[start:start+l,start:start+l]=np.ones((l,l))*(1/l)
        start+=l
    O=np.ones((n,n))*(1/n)    
    Hb=K.dot(B-O)
    Ht=K.dot(np.eye(n)-O)

    Ut,convt,_=svd(Ht,full_matrices=False)
    inv_convt=inv(np.diag(convt))
    T=inv_convt.dot(Ut.T).dot(Hb)
    P,*_=svd(T,full_matrices=False)
    G=Ut.dot(inv_convt).dot(P)
    A,_=qr(G)
    
    return A,func