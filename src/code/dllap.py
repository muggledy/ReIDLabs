'''
See demo in ../demo_laplacianl2_kcca_viper.py
muggledy 2020/6/7
'''

import numpy as np
from scipy.linalg import solve_sylvester
from .graph_sc import learn_dict
from itertools import count
from .cprint import cprint_err,cprint_out

def opt_lap_Y(X,D,L,alpha,beta):
    '''min ||X-DY||_F^2 + alpha||Y||_F^2 + beta*tr(YLYᵀ)'''
    return solve_sylvester(D.T@D+beta*np.eye(D.shape[1]),alpha*L,D.T@X)

def dllap(X,W,nBasis,alpha,beta,max_iter=50,eps=0.001,**kwargs):
    '''min ||X-DY||_F^2 + alpha||Y||_F^2 + beta*tr(YLYᵀ) s.t. ||D_i||^2<=1'''
    L=np.diag(np.sum(W,axis=1))-W
    #init D
    # D=np.random.rand(X.shape[0],nBasis)-0.5
    # D-=np.mean(D,axis=0)
    # D=D@np.diag(1/np.sqrt(np.sum(D**2,axis=0))) #same as source code, not good
    D=np.random.rand(X.shape[0],nBasis) #good. But randn is not 
                                        #recommended, because Newton in 
                                        #learn_dict may always not convergenc in 
                                        #200 iters(also is rand), however, the 
                                        #last rank scores maybe right
    if kwargs.get('Dinit') is not None:
        D=kwargs.get('Dinit') #use old D as initial can reduct iterations of Newton!
                              #but may induce some bias: fobj increment
    calc_obj=lambda X,Y,D,L,alpha,beta: \
        np.sum((X-D@Y)**2)+alpha*np.sum(Y**2)+beta*np.trace(Y@L@(Y.T))
    old_obj=None
    for i in count():
        if i>=max_iter:
            cprint_err('(DLLAP)max iter(%d)!'%max_iter)
            break
        #fix D,update Y
        Y=opt_lap_Y(X,D,L,alpha,beta)
        #fix Y,update D
        D=learn_dict(X,Y,max_iter=200,info=True)
        #calc obj value
        obj=calc_obj(X,Y,D,L,alpha,beta)
        print('epoch=%d,fobj:%f'%(i+1,obj))
        if old_obj is not None:
            if np.abs(old_obj-obj)<eps:
                cprint_out('(DLLAP)convergence(%d)!'%(i+1))
                break
        old_obj=obj
    return D