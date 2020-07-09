from scipy.linalg import logm
import numpy as np
import tensorflow as tf

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

def logm1(X):
    '''calculate log of matrix with tensorflow-gpu, very fast, takes 
       only 200s on VIPeR now'''
    d=X.shape[-1]
    X=X+np.eye(d)*0.01
    real=tf.convert_to_tensor(X,dtype=tf.float32)
    img=tf.convert_to_tensor(np.zeros(X.shape),dtype=tf.float32)
    data=tf.complex(real,img)
    return tf.math.real(tf.linalg.logm(data)).numpy()
