'''
See demo in ../../demo/kcca_dllap2_viper.py
muggledy 2020/6/7
'''

import numpy as np
from scipy.linalg import solve_sylvester
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from zoo.graph_sc import learn_dict
from itertools import count
from zoo.tools import sym_matric,cosine_dist
from zoo.cprint import cprint_err,cprint_out

def max_knn(D,k):
    '''salient D's maximum k elements according to axis 1(horizon)'''
    ret=np.zeros(D.shape)
    sort_inds=np.argsort(D,axis=1)[:,::-1]
    sal_inds=(np.broadcast_to(np.arange(D.shape[0])[:,None],(D.shape[0],k)),sort_inds[:,:k])
    ret[sal_inds]=D[sal_inds]
    return ret

def get_cross_view_graph(*X,k=5,**kwargs):
    '''construct soft cross-view correspondence relationship matric W, more precisely, if x_i is 
       among the k-nearest neighbours of y_j or vice versa, W_{i,j}=cosine_dist(x_i,y_j), else 
       w_{i,j}=0. X is a set, each of which represents all samples coming from one camera'''
    n=len(X)
    y,x=np.meshgrid(range(n),range(n))
    inds=np.triu_indices(n,1)
    x,y=x[inds],y[inds]
    t=np.cumsum([0]+[i.shape[1] for i in X])
    W=np.zeros((t[-1],t[-1])) #we don't consider intra-view relationship, just set them as zero
    for i,j in zip(x,y):
       simi=1-cosine_dist(X[i],X[j])
       d2=max_knn(simi.T,k)
       if kwargs.get('backdoor'):
           print('use backdoor!')
           d1=d2 #same as source code of laplacianL2, but it is 
                 #not reasonable and may cause shape mismatch error!
       else:
           d1=max_knn(simi,k) #this(I) should be right!
       W[t[i]:t[i+1],t[j]:t[j+1]]=(d1+d2.T)/2
    W=sym_matric(W)
    W[(W<0.5)&(W!=0)]=0.0001 #remove "noise" from the graph
    W[range(W.shape[0]),range(W.shape[0])]=0
    return W

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