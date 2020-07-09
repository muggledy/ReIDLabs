import numpy as np
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from zoo.tools import euc_dist,norm_labels
from zoo.cprint import cprint_out
from functools import reduce,partial

def kernel_gaussian(X,Y=None,**kwargs):
    '''calc kernel matric of X(d,n1) and Y(d,n2) with gaussian kernel. Y will be the same 
       as X if Y==None defaulty'''
    sigma=kwargs.get('sigma',1) #in fact, this super parameter may be different with 
                                #different dataset, sigma is set as 1 for viper, 2 for
                                #prid2011, 0.5 for cuhk01 and cuhk03 in paper[1]
    Y=X if Y is None else Y
    D=euc_dist(X,Y)
    D=(-D)/(2*sigma**2)
    return np.exp(D)

def kernel_linear(X,Y=None):
    '''linear kernel function'''
    Y=X if Y is None else Y
    return X.T.dot(Y)

def get_kernel_Hw_Hb_Ht(X,labels,kernel='gaussian',**kwargs):
    labels=norm_labels(labels)
    sortedInd=np.argsort(labels)
    labels=labels[sortedInd]
    X=X[:,sortedInd]
    groups=reduce \
        (lambda x,y:x+[[y]] if x[-1][-1]!=y else x[:-1]+[x[-1]+[y]],labels,[['^']])[1:]

    cprint_out('using %s kernel'%kernel)
    if kernel=='gaussian': #RBF
        K=kernel_gaussian(X,None,**kwargs)
        func=partial(kernel_gaussian,X,**kwargs)
    elif kernel=='sigmoid':
        pass
    elif kernel=='linear':
        K=kernel_linear(X,None)
        func=partial(kernel_linear,X)
    else:
        raise ValueError('invalid kernel function!')
    
    _,n=X.shape
    B=np.zeros((n,n))
    start=0
    for g in groups:
        l=len(g)
        B[start:start+l,start:start+l]=np.ones((l,l))*(1/l)
        start+=l
    O=np.ones((n,n))*(1/n)
    I=np.eye(n)
    Hw=K.dot(I-B)
    Hb=K.dot(B-O)
    Ht=K.dot(I-O)
    return Hw,Hb,Ht,func

def klda(X,labels,n_components=None,regularizer=None,kernel='gaussian',**kwargs):
    '''kernel version of LDA'''
    Hw,Hb,_,func=get_kernel_Hw_Hb_Ht(X,labels,kernel,**kwargs)
    Sw=Hw.dot(Hw.T)
    Sb=Hb.dot(Hb.T)

    c=len(np.unique(labels))
    eigVals,eigVects=np.linalg.eig(np.linalg.inv(Sw+(np.eye(Sw.shape[0])*regularizer if regularizer!=None else 0)).dot(Sb))
    maxSortedInd=np.argsort(eigVals)[::-1][:c-1 if n_components==None else n_components]
    A=eigVects[:,maxSortedInd]
    return A,func

if __name__=='__main__':
    import scipy.io as scio
    import os.path
    import matplotlib.pyplot as plt
    from zoo.lda import lda
    data=scio.loadmat(os.path.join(os.path.dirname(__file__),'../../data/lda_data_demo.mat'))
    train_data=data['train_c1'],data['train_c2'],data['train_c3']
    test_data=data['test_c1'],data['test_c2'],data['test_c3']
    markers=['.','x','+']
    colors=['red','green','blue']
    plt.subplot(131)
    for i,s in enumerate(test_data):
        plt.scatter(*(s.T),marker=markers[i],color=colors[i])
    plt.xlabel('original data')
    L=lda(np.vstack(train_data).T,[0,]*len(train_data[0])+[1,]*len(train_data[1])+[2,]*len(train_data[2]),1)
    plt.subplot(132)
    for i,s in enumerate(test_data):
        x=L.T.dot(s.T)
        plt.scatter(x,np.ones(x.shape)*-7,marker=markers[i],color=colors[i])
    plt.xlabel('lda')
    
    L,K=klda(np.vstack(train_data).T,[0,]*len(train_data[0])+[1,]*len(train_data[1])+[2,]*len(train_data[2]),1,regularizer=0.001,kernel='gaussian',sigma=0.6)
    plt.subplot(133)
    for i,s in enumerate(test_data):
        x=L.T.dot(K(s.T))
        plt.scatter(x,np.ones(x.shape)*-7,marker=markers[i],color=colors[i])
    plt.xlabel('klda(gaussian)')
    plt.axis('equal')
    plt.show()