'''
References:
[1] Kodirov E., Xiang T., Fu Z., Gong S. (2016) Person Re-Identification by Unsupervised ℓ1 Graph 
    Learning. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 
    2016. Lecture Notes in Computer Science, vol 9905. Springer, Cham
muggledy 2020/5/4
'''

import numpy as np
from .tools import cosine_dist,sym_matric,norm1_dist
from .optimize import opt_dict,opt_soft_threshold
from .cprint import cprint_out,cprint_err
from itertools import count
from functools import reduce
from scipy.linalg import solve_sylvester
from scipy.stats import ortho_group

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
    W=np.zeros((t[-1],t[-1]))
    for i,j in zip(x,y):
       simi=1-cosine_dist(X[i],X[j])
       d2=max_knn(simi.T,k)
       if kwargs.get('backdoor')==True:
           d1=d2 #same as source code of laplacianL2, but it is not reasonable
       else:
           d1=max_knn(simi,k) #this should be right! but result(rank1 of cmc) may decline
       W[t[i]:t[i+1],t[j]:t[j+1]]=(d1+d2.T)/2
    W=sym_matric(W) #we don't consider intra-view relationship, just set them as zero
    W[(W<0.5)&(W!=0)]=0.0001 #remove noise from the graph
    W[range(W.shape[0]),range(W.shape[0])]=0
    return W

def opt_W_lambd2(Y,k,lambd2):
    '''update W: min sum_{i,j}W_{i,j}||y_i-y_j||_1+λ2||W||_F^2 s.t. W_i^T1=1,W_{i,j}>=0'''
    d=norm1_dist(Y,Y)/(2*lambd2)
    d_=np.sort(d,axis=0)
    W=np.maximum(((1+np.sum(d_[:k,:],axis=0))/k)[None,:]-d,0)
    #W=sym_matric(W)
    W=(W+W.T)/2
    lambd2=np.sum((k/2)*d[k,:]-(0.5*np.sum(d[:k,:],axis=1)[:,None]))/(Y.shape[1])
    return W,lambd2

def calc_Aw(W):
    Lw=np.diag(np.sum(W,axis=1))-W
    Sw,Uw=np.linalg.eig(Lw) #Lw=Uw·Sw·Uwᵀ
    Aw=Uw.dot(np.diag(np.sqrt(Sw)))
    return Aw

def udlgl(*X,k,lambd1,lambd2,gamma,nBasis,rho):
    '''Unsupervised ℓ1 Graph dict Learning'''
    X=[i.astype('float32') for i in X]
    n=reduce(lambda x,y:x+y,[int(i.shape[1]) for i in X]) #num of samples from all cameras
    d=X[0].shape[0] #dim of sample
    W=get_cross_view_graph(*X,k=k)
    nBasis=2**8 if nBasis==None else nBasis
    Y=np.random.randn(nBasis,n)
    Aw=calc_Aw(W)
    U=Y.dot(Aw)
    F=np.random.randn(Y.shape[0],Aw.shape[1])
    max_iter=50
    for i in count():
       if i>=max_iter:
          cprint_err('Udlgl max iter(%d)!'%max_iter)
          break
       print('update D...')
       D=opt_dict(np.hstack(X),Y)
       print('update Y...')
       Y=solve_sylvester(np.nan_to_num(D.T.dot(D)),np.nan_to_num(gamma*Aw.dot(Aw.T)),np.nan_to_num(D.T.dot(np.hstack(X))+gamma*U.dot(Aw.T)+F.dot(Aw.T))) #np.nan_to_num
       print('update U...')
       U=opt_soft_threshold(Y.dot(Aw)-F/gamma,lambd1/gamma)
       print('update W...')
       W,lambd2=opt_W_lambd2(Y,k,lambd2)
       print(U.dtype,Y.dtype,Aw.dtype)
       F+=(gamma*(U.real-Y.real.dot(Aw.real)))
       gamma*=rho
       Aw=calc_Aw(W)
    return D