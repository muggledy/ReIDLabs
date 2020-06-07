'''
References:
[1] Kodirov E., Xiang T., Fu Z., Gong S. (2016) Person Re-Identification by Unsupervised ℓ1 Graph 
    Learning. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 
    2016. Lecture Notes in Computer Science, vol 9905. Springer, Cham
muggledy 2020/5/4
'''

import numpy as np
from .tools import cosine_dist,sym_matric,norm1_dist
from .optimize import opt_soft_threshold
from .cprint import cprint_out,cprint_err
from .graph_sc import learn_dict
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
       if kwargs.get('backdoor'):
           print('enter backdoor!')
           d1=d2 #same as source code of laplacianL2, but it is 
                 #not reasonable and may cause shape mismatch error!
       else:
           d1=max_knn(simi,k) #this(I) should be right!
       W[t[i]:t[i+1],t[j]:t[j+1]]=(d1+d2.T)/2
    W=sym_matric(W) #we don't consider intra-view relationship, just set them as zero
    W[(W<0.5)&(W!=0)]=0.0001 #remove noise from the graph
    #W[range(W.shape[0]),range(W.shape[0])]=0
    return W

def opt_W_lambd2(Y,k,lambd2):
    '''update W: min sum_{i,j}W_{i,j}||y_i-y_j||_1+λ2||W||_F^2 s.t. W_i^T1=1,W_{i,j}>=0'''
    d=norm1_dist(Y,Y)/(2*lambd2)
    d_=np.sort(d,axis=0)
    W=np.maximum(((1+np.sum(d_[:k,:],axis=0))/k)[None,:]-d,0)
    #W=sym_matric(W)
    W=(W+W.T)/2
    lambd2=np.sum((k/2)*d[k,:]-(0.5*np.sum(d[:k,:],axis=1)[:,None]))/(Y.shape[1])
    #lambd2=np.sum((k/2)*d[:,[k]]-(0.5*np.sum(d[:,:k],axis=1)[:,None]))/(Y.shape[1])
    return W,lambd2

def calc_Aw(W):
    Lw=np.diag(np.sum(W,axis=1))-W
    Sw,Uw=np.linalg.eig(Lw) #Lw=Uw·Sw·Uwᵀ
    #Aw=Uw.dot(np.diag(np.sqrt(Sw))) #sqrt will induce complex!
    Aw=Uw.dot(np.diag(np.sqrt(np.abs(Sw))))
    return Aw

def calc_obj(X,D,Y,U,F,Aw,W,lambd1,lambd2,gamma):
    return 0.5*np.sum((X-D.dot(Y))**2)+lambd1*np.sum(np.abs(U))+ \
       np.trace(F.T.dot(U-Y.dot(Aw)))+gamma/2*np.sum((U-Y.dot(Aw))**2)+lambd2*np.sum(W**2)

def udlgl(*X,k,lambd1,lambd2,gamma,nBasis,rho,max_iter,W_init=None):
    '''Unsupervised ℓ1 Graph dict Learning'''
    X=[i.astype('float32') for i in X]
    n=sum([int(i.shape[1]) for i in X]) #num of samples from all cameras
    W=get_cross_view_graph(*X,k=k,backdoor=True) if W_init is None else W_init
    X=np.hstack(X)
    Aw=calc_Aw(W)
    nBasis=2**9 if nBasis==None else nBasis
    Y=np.random.rand(nBasis,n)
    U=np.random.rand(nBasis,Aw.shape[1])
    #U=Y.dot(Aw)
    F=np.random.rand(nBasis,Aw.shape[1])
    for i in count():
       if i>=max_iter:
          cprint_err('Udlgl max iter(%d)!'%max_iter)
          break
       print('update D...',end='\r')
       D=learn_dict(X,Y)
       print('update Y...',end='\r')
       Y=solve_sylvester(D.T@D,gamma*Aw@(Aw.T),D.T@X+gamma*U@(Aw.T)+F@(Aw.T))
       print('update U...',end='\r')
       U=opt_soft_threshold(Y@Aw-F/gamma,lambd1/gamma)
       print('update W...',end='\r')
       W,_=opt_W_lambd2(Y,k,lambd2)
       Aw=calc_Aw(W)
       F+=(gamma*(U-Y.dot(Aw)))
       gamma*=rho
       print('fobj:',calc_obj(X,D,Y,U,F,Aw,W,lambd1,lambd2,gamma))
    return D