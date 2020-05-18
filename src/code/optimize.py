'''
see doc in ./optimize.md
muggledy 2020/5/8
'''

import numpy as np
from numpy.linalg import inv
from itertools import count
from .cprint import cprint_out,cprint_err

def opt_dict(X,Y,c=1):
    '''update D: min ||X-DY||_F^2 s.t. ||D(:,i)||_2^2<=c'''
    k=Y.shape[0]
    dual_lambds=np.abs(np.random.rand(k,1)) #any arbitrary initialization should be ok
    ### update dual_lambds
    XY_T=X.dot(Y.T)
    YY_T=Y.dot(Y.T)
    max_iter=100
    eps=0.0001
    for i in count():
        if i>=max_iter:
            cprint_err('Newton max iter(%d)!'%max_iter)
            break
        YY_T_inv=inv(YY_T+np.diag(dual_lambds.reshape(-1)))
        gradient=(np.sum((XY_T.dot(YY_T_inv))**2,axis=0)-c)[:,None]
        t=XY_T.dot(YY_T_inv)
        hessian=-2*(t.T.dot(t)*YY_T_inv)
        old_dual_lambds=dual_lambds
        dual_lambds=dual_lambds-inv(hessian+0.001*np.eye(hessian.shape[0])).dot(gradient) #
        if np.sum((dual_lambds-old_dual_lambds)**2)<eps:
            cprint_out('Newton convergence(%d)!'%i)
            break
    ### by Newton's method
    return XY_T.dot(YY_T_inv)

def opt_soft_threshold(B,lambd):
    '''update X: min ||X-B||_2^2+2λ||X||_1'''
    return np.sign(B)*np.maximum(np.abs(B)-lambd,0)

def opt_coding_l2(D,X,lambd):
    '''update y: min ||x-Dy||_F^2+λ||y||_2^2'''
    P=inv(D.T.dot(D)+lambd*np.eye(D.shape[1])).dot(D.T)
    Y=P.dot(X)
    return Y