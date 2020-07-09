'''
References:
[1] M. Zheng et al., "Graph Regularized Sparse Coding for Image Representation," in IEEE 
    Transactions on Image Processing, vol. 20, no. 5, pp. 1327-1336, May 2011, doi: 
    10.1109/TIP.2010.2090535.
muggledy 2020/5/20
'''

import numpy as np
from numpy.linalg import inv,pinv,matrix_rank
from itertools import count
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from zoo.cprint import cprint_out,cprint_err

def learn_dict(X,Y,c=1,max_iter=100,info=False):
    '''update D: min ||X-DY||_F^2 s.t. ||D(:,i)||_2^2<=c'''
    k=Y.shape[0]
    dual_lambds=np.abs(np.random.randn(k,1)) #any arbitrary initialization should be ok
    ### update dual_lambds
    XY_T=X.dot(Y.T)
    YY_T=Y.dot(Y.T)
    for i in count():
        if i>=max_iter:
            if info:
                cprint_err('Newton Max Iter(%d)!'%max_iter)
            break
        # YY_T_inv=pinv(YY_T+np.diag(dual_lambds.reshape(-1))) #pinv may raise error(numpy.linalg.LinAlgError: 
        # SVD did not converge) in DLLAP, i don't know why
        YY_T_inv=inv(YY_T+np.diag(dual_lambds.reshape(-1))+0.0001)
        gradient=(np.sum((XY_T.dot(YY_T_inv))**2,axis=0)-c)[:,None]
        t=XY_T.dot(YY_T_inv)
        hessian=-2*(t.T.dot(t)*YY_T_inv)
        old_dual_lambds=dual_lambds
        # dual_lambds=dual_lambds-pinv(hessian).dot(gradient) #numpy.linalg.LinAlgError: SVD did not converge
        dual_lambds=dual_lambds-inv(hessian+0.0001*np.eye(hessian.shape[0])).dot(gradient)
        #eps=0.0001
        #if np.sum((dual_lambds-old_dual_lambds)**2)<eps:
        if np.allclose(dual_lambds,old_dual_lambds):
            if info:
                cprint_out('Newton Convergence(%d)!'%(i+1))
            break
    ### by Newton's method
    return XY_T.dot(YY_T_inv)

########################################################################################
# who can write feature sign search(as same as source code)?
# see http://www.cad.zju.edu.cn/home/dengcai/Data/SparseCoding.html