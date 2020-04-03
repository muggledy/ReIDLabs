'''
References:
[1] Dai J , Zhang Y , Lu H , et al. Cross-view Semantic Projection Learning for Person 
    Re-identification[J]. Pattern Recognition, 2017:S0031320317301723.
'''

import numpy as np
from itertools import count
from numpy.linalg import inv
from .lomo.tools import measure_time

@measure_time
def cspl(X1,X2,k=None,mu=1,beta=1,lambda_mu=0.05,lambda_v=0.2,lambda_a=0.2,lambda_p=0.2,iter=None):
    d,n=X1.shape
    k=d if k==None else k
    eps=0.1
    
    #np.random.seed(0)
    U=np.random.randn(d,k) #如果不降低样本维数，譬如lomo特征立马会爆内存，导致死机，因为通过计算，
                           #假设浮点数以64位存储的话，此处预分配的6个矩阵需要大约16G内存用量。但论
                           #文中并无降维步骤，viper lomo特征从26960降维到600维后仅需要约13M内存
    V1=np.random.randn(k,n)
    V2=np.random.randn(k,n)
    P1=np.random.randn(k,d)
    P2=np.random.randn(k,d)
    A=np.eye(k)
    #A=np.random.randn(k,k)
    
    for i in count():
        print('iter %d'%i,end='\r')
        if iter!=None and i>=iter:
            print('cspl max iteration')
            break
        old_U=U #成倍的内存
        U=(X1.dot(V1.T)+X2.dot(V2.T)).dot(inv(V1.dot(V1.T)+V2.dot(V2.T)+lambda_mu*np.eye(k)))
        old_V1=V1
        V1=inv(U.T.dot(U)+(mu+beta+lambda_v)*np.eye(k)).dot(U.T.dot(X1)+beta*A.dot(V2)+mu*P1.dot(X1))
        old_V2=V2
        V2=inv(U.T.dot(U)+beta*A.T.dot(A)+(mu+lambda_v)*np.eye(k)).dot(U.T.dot(X2)+beta*A.T.dot(V1)+mu*P2.dot(X2))
        old_P1=P1
        P1=V1.dot(X1.T).dot(inv(X1.dot(X1.T)+(lambda_p/mu)*np.eye(d)))
        old_P2=P2
        P2=V2.dot(X2.T).dot(inv(X2.dot(X2.T)+(lambda_p/mu)*np.eye(d)))
        old_A=A
        A=V1.dot(V2.T).dot(inv(V2.dot(V2.T)+(lambda_a/beta)*np.eye(k)))
        
        if np.sum(np.abs(U-old_U))<eps and np.sum(np.abs(V1-old_V1))<eps and np.sum(np.abs(V2-old_V2))<eps \
                           and np.sum(np.abs(P1-old_P1))<eps and np.sum(np.abs(P2-old_P2))<eps \
                           and np.sum(np.abs(A-old_A))<eps:
            print('cspl convergence(%d iters)'%i)
            break
    
    return P1,P2,A
