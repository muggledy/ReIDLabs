import numpy as np
import cv2
from .lomo.tools import calc_cmc
from functools import reduce

def euc_dist(X,Y):
    '''calc euclidean distance of X(d*m) and Y(d*n), func return 
       a dist matrix D(m*n), D[i,j] represents the distance of X[:,i] 
       and Y[:,j]'''
    A=np.sum(X.T*(X.T),axis=1)
    D=np.sum(Y.T*(Y.T),axis=1)
    return A[:,None]+D[None,:]-X.T.dot(Y)*2

def norm_labels(labels):
    '''make category labels grow from 0 one by one, for example: 
       [1 7 2 2 2 3 5 1] -> [0 4 1 1 1 2 3 0]'''
    labels=np.array(labels)
    t=np.arange(np.max(labels)+1)
    if len(t)==len(labels) and np.sum(labels-t)==0:
        return labels
    ret=labels.copy()
    s=np.sort(np.unique(labels))
    for i,e in enumerate(s):
        ret[np.where(labels==e)]=i
    return ret

def shuffle_before_cmc(dist,ylabels,xlabels,seed=None): #deprecated
    '''dist's horizonital axis corresponds to ylabels and vertical axis 
       corresponds to xlabels. we will shuffle the order of probe and 
       gallery, and normalize the labels simultaneously because function 
       calc_cmc in lomo.tools.py doesn't normalize the labels(it may 
       cause index error), so it's better to do this op before calc cmc'''
    yn=np.array(ylabels).shape[0]
    xn=np.array(xlabels).shape[0]
    if dist.shape!=(xn,yn):
        raise ValueError \
            ('dist\'s shape must be equal to (len(xlabels),len(ylabels))!')
    if seed!=None:
        np.random.seed(seed) #shuffle samely with the same seed
    ind=np.arange(xn)
    np.random.shuffle(ind)
    dist=dist[ind,:]
    xlabels=norm_labels(xlabels[ind])
    ind=np.arange(yn)
    np.random.shuffle(ind)
    dist=dist[:,ind]
    ylabels=norm_labels(ylabels[ind])
    return dist,ylabels,xlabels

def range2(start,end,step):
    cur=start
    while cur<=end:
        yield cur
        cur+=step

def gen_program(n):
    program='def gen_01(step=0.1):\n    s1=0\n'
    for i in range(1,n):
        program+=('    '*i+'for i%d in range2(0,1-s%d,step):\n'%(i,i)
            +'    '*(i+1)+'s%d=s%d+i%d\n'%(i+1,i,i))
    program+=('    '*n+'yield (%s1-s%d)'%( \
            str(''.join(['i%d,'%i for i in range(1,n)])),n))
    exec(program,globals())

def seek_good_coeffi(dists,pLabels,gLabels,numRank=100,step=0.01):
    '''dists' horizonital axis indicates probe and ..., coefficients 
       for different dist matrix are trade-off, range from (0,1)'''
    n=len(dists)
    gen_program(n)
    max_rank1=0
    max_coeffi=None
    for i,e in enumerate(gen_01(step)):
        if i%100==0:
            print(i,end='\r')
        rank1=calc_cmc(reduce(lambda x,y:x+y, \
            [xe*dists[xi] for xi,xe in enumerate(e)]), \
                pLabels,gLabels,numRank)[0]
        if rank1>max_rank1:
            max_rank1=rank1
            max_coeffi=e
    return max_coeffi,max_rank1

def create_ellipse_mask(shape):
    '''create an ellipse mask(one channel) with arg shape(h,w), h//2 
       and w//2 represent the length of ellipse's two axises'''
    h,w=shape
    ret=np.zeros(shape)
    cv2.ellipse(ret,(w//2,h//2),(w//2,h//2),0,0,360,(255,255,255),-1)
    return ret.astype('uint8')

def sym_matric(x):
    '''create a randn symmetric matric defaultly with shape(x,x). Or 
       you can give half data(x, a list) with length (1+n)*n/2, also 
       you can pass a square matric, we will copy its upper triangle'''
    if isinstance(x,int):
        M=np.random.randn(x,x)
    elif isinstance(x,list):
        t=np.sqrt(len(x)*2)
        n=int(np.floor(t))
        if n*(n+1)/2!=len(x):
            raise ValueError \
                ('x\'s length must be subjected to (1+n)*n/2=len(x)!')
        else:
            M=np.zeros((n,n))
            indx,indy=np.triu_indices(n)
            M[indx,indy]=x
    elif isinstance(x,np.ndarray) and len(x.shape)==2 \
                  and x.shape[0]==x.shape[1]:
        M=x
    else:
        raise ValueError('x must be square matric!')
    Mt=np.triu(M)
    Mt+=Mt.T-np.diag(Mt.diagonal())
    return Mt
