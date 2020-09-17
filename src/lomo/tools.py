import numpy as np
import matplotlib.pyplot as plt
from functools import wraps,reduce
import time
from itertools import count
import os.path
import traceback

def measure_time(wrapped):
    @wraps(wrapped)
    def wrapper(*args,**kwds):
        t1=time.time()
        ret=wrapped(*args,**kwds)
        t2=time.time()
        print('@measure_time: {0} took {1} seconds'.format(wrapped.__name__,t2-t1))
        return ret
    return wrapper

def getcwd(file=None):
    if file==None:
        t=traceback.extract_stack()
        return os.path.dirname(os.path.realpath(t[0].filename))
    elif not os.path.exists(str(file)):
        file=os.path.realpath(__file__)
    return os.path.dirname(os.path.realpath(file))

def normc(a,axis=0):
    '''normalize col(axis=0) or row(axis=1) to unit length'''
    s=np.sqrt(np.sum(a*a,axis=axis))
    if axis==1:
        s=s[:,None]
    return a/(s+np.finfo(np.float32).eps)

def mah_dist(M,X,Y):
    '''return dist matrix of X and Y, col represents X and row represents Y'''
    u=np.sum(X.T.dot(M)*(X.T),axis=1)
    v=np.sum(Y.T.dot(M)*(Y.T),axis=1)
    return u[:,None]+v[None,:]-X.T.dot(M).dot(Y)*2

def cmc(D,P,G,rank=None):
    '''suitable for single-shot, please always call calc_cmc()'''
    h,w=D.shape
    s=np.argsort(D,axis=0)
    s=G[s]
    match=(s==P)
    C=np.sum(match,axis=1)/w
    rank=np.min([h if rank==None else rank,h])
    return np.cumsum(C)[:rank]

def calc_cmc(D,P,G,rank=None):
    '''D's col represents gallery and row represents probe. see github or refer 
       to https://www.cnblogs.com/xiaoaoran/p/10854911.html'''
    if isinstance(P,int) and isinstance(G,int): #P,G can be integer
        P=np.arange(P)
        G=np.arange(G)
    elif not (isinstance(P,np.ndarray) and isinstance(G,np.ndarray)):
        P=np.array(P)
        G=np.array(G)
    nptypes=np.unique(P)
    if len(nptypes)!=len(P):
        print('WARNNING(CMC): Identities in Probe must be unique!')
    ngtypes=np.unique(G)
    if not (len(nptypes)==len(ngtypes) and np.sum(np.sort(nptypes)-np.sort(ngtypes))==0):
        print \
        ('WARNNING(CMC): The type of identities in Probe and Gallery should be equal!')
    if D.shape[1]!=len(P):
        raise ValueError('D\'s width must be equal to P\'s length!')
    if len(ngtypes)==len(G):
        if len(G)!=D.shape[0]:
            raise ValueError \
            ('D\'s height must be equal to G\'s length in Single-Shot Problem!')
        #print('Single-Shot')
        return cmc(D,P,G,rank)
    #print('Multi-Shot')
    ginds={}
    for t in ngtypes:
        ginds[t]=np.where(G==t)[0]
    ntimes=reduce(lambda x,y:x*y,[len(i) for i in ginds.values()])
    ntimes=100 if ntimes>100 else ntimes
    c=[]
    for i in range(ntimes):
        g,gind=[],[]
        for gk,gv in ginds.items():
            g.append(gk)
            gind.append(np.random.choice(gv))
        c.append(cmc(D[gind,:],np.array(P),np.array(g),rank))
    return np.mean(np.array(c),axis=0)

def plot_cmc(c,labels,density=[1,2,5,8,10,20,50,80,100,150,],verbose=False):
    '''see github'''
    x=np.arange(1,c.shape[-1]+1)
    c=c*100
    lines=plt.plot(x,c.T)
    plt.legend(labels)
    plt.grid(which='both',linestyle=':')
    t=np.sort(c.flatten())
    if density:
        rank=len(x)
        den=np.array([i for i in density if i<rank]+[rank])
        plt.xticks(x[den-1])
        plt.yticks(np.sort(c[...,den-1].ravel()))
        if verbose:
            if len(c.shape)==1:
                c=c.reshape(1,-1)
            for p,i in enumerate(c):
                color=lines[p].get_c()
                plt.plot(den,c[p,den-1],marker='^',linestyle='',color=color)
                for k,j in enumerate(i[den-1]):
                    plt.annotate('%.2f'%j,(den[k]+0.5,j-1.5),color=color,fontsize=8)
    elif density==None:
        plt.xticks(x)
        plt.yticks(t)
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Matching Score(%)')
    plt.show()

if __name__=='__main__':
    pass