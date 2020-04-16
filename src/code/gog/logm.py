from scipy.linalg import logm
import numpy as np
from functools import reduce

def logm0(X):
    '''X's shape is (...,d,d), i.e. X's last two dimension must be 
       square, for these sub-matrix, they must be non-singular, so 
       i add a identity matrix with a small positive constant'''
    shape=X.shape
    X=X.reshape(-1,*(shape[-2:]))
    ret=np.zeros(X.shape)
    for i in range(X.shape[0]):
        ret[i,:,:]=logm(X[i,:,:]+0.001*np.eye(shape[-1])) #if i can 
                                #find a C/C++ version of logm, i can 
                                #accelerate it with Cython. 
                                #Unfortunately, this way spends almost 
                                #8 hours on VIPeR now!
    return ret.reshape(shape)

'''ERROR
def logm1(X):
    shape=X.shape
    nums=reduce(lambda x,y:x*y,shape)
    n=int(np.ceil(np.sqrt(nums)))
    X=np.hstack((X.ravel(),np.zeros((n**2-nums,)))).reshape(n,n)
    ret=logm(X+0.001*np.eye(n)).ravel()[:nums]
    return ret.real.reshape(shape)
'''