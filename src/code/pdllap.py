import numpy as np
from .camel import X_data,construct_X_,construct_cov_,construct_D,construct_U_
from .graph_sc import learn_dict
from itertools import count
from scipy.linalg import solve_sylvester,eig,inv,pinv
from .cprint import cprint_err
import tensorflow as tf

from importlib import reload #
import os #
import scipy.io as scio #
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'./matlab/tools/'))

def decompose_U_(U_,n):
    '''U_=[U_1^T,...,U_n^T]^T, U has shape (origin_dim,proj_dim)'''
    return np.split(U_,n)

def construct_U_without_cov_(eigV,eigU,nBasis):
    inds=np.argsort(eigV)[:nBasis]
    us_unnormalized=eigU[:,inds]
    U_=np.zeros(us_unnormalized.shape)
    for i in range(U_.shape[1]):
        u_un=us_unnormalized[:,[i]]
        U_[:,[i]]=u_un/np.sqrt(u_un.T@u_un)
    return U_

def tf_eig(X):
    X=tf.convert_to_tensor(X,dtype=tf.float32)
    eigV,eigU=tf.linalg.eigh(X)
    return eigV.numpy(),eigU.numpy()

def tf_inv(X):
    X=tf.convert_to_tensor(X,dtype=tf.float32)
    return tf.linalg.inv(X).numpy()

def pdllap(data,id_views,W,k_nn,nBasis,pjDim,lambd1,lambd2,lambd3,lambd4,max_iter=50,use_cov=False):
    '''ptoject(subspace) + DLLAP
       min ||UᵀX-DY||_F^2 + λ1*tr(UᵀSU) + λ2||Y||_F^2 + λ3*tr(YLYᵀ) + λ4||W||_F^2
       s.t. ||D_i||_2^2<=1, Uᵀ∑U=VI, W_i^T1=1, W_i>=0'''
    X=X_data()
    X.initialize_data(data,id_views)
    X_=construct_X_(X)
    V=X.info['num_cameras']
    dim=X.info['dim_sample']
    U_=np.random.rand(V*dim,pjDim)-0.5
    U_-=np.mean(U_,axis=0)
    U_=U_@np.diag(1/np.sqrt(np.sum(U_**2,axis=0)))

    D=np.random.rand(pjDim,nBasis)-0.5
    D-=np.mean(D,axis=0)
    D=D@np.diag(1/np.sqrt(np.sum(D**2,axis=0)))

    L=np.diag(np.sum(W,axis=1))-W
    if use_cov:
        cov_=construct_cov_(X)
        cov_inv=tf_inv(cov_) #consume much time
    S=construct_D(X)
    X_X_T=X_@(X_.T)
    for i in count():
        if i>=max_iter:
            cprint_err('(PDLLAP)max iter(%d)!'%max_iter)
            break
        U_TX_=U_.T@X_
        #update Y
        print('update Y...',end='\r')
        Y_=solve_sylvester(D.T@D+lambd2*np.eye(D.shape[1]),lambd3*L,D.T@U_TX_)
        #update D
        print('update D...',end='\r')
        # D=learn_dict(U_TX_,Y_,max_iter=100,info=True)

        scio.savemat(os.path.join(os.path.dirname(__file__),'../../data/XS.mat'),{'X':U_TX_,'S':Y_})
        os.chdir(os.path.join(os.path.dirname(__file__),'./matlab/tools/'))
        if i==0:
            import test_demo
        else:
            reload(test_demo)
        D=scio.loadmat(os.path.join(os.path.dirname(__file__),'../../data/XS_D.mat'))['D']

        os.chdir(os.path.join(os.path.dirname(__file__),'../'))


        #update U
        print('update U...',end='\r')
        A=pinv(U_TX_)@D
        t=(cov_inv if use_cov else np.eye(X_X_T.shape[0]))@(X_X_T-X_@(Y_.T)@(A.T)@(X_.T)-X_@A@Y_@(X_.T)+X_@A@(Y_@(Y_.T))@(A.T)@(X_.T)+lambd1*S)
        eigV,eigU=tf_eig(t)
        U_=construct_U_(eigV,eigU,nBasis,cov_,V) if use_cov else \
                                                construct_U_without_cov_(eigV,eigU,nBasis)
        #update W
        #...
    return D,decompose_U_(U_,V)