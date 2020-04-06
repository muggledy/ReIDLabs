import numpy as np

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
