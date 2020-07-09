import time
import numpy as np
cimport numpy as np

#本文件编写参考：https://blog.csdn.net/qq_33353186/article/details/80298239

np.import_array()

#作用相当于c++头文件
cdef extern from "main.h":
	void batch_omp_wrapper(double *_alpha0,double *_G,double *_result,int aL,int Gn,int Gm,double e0,double err,int k)

def omp(np.ndarray[double,ndim=2,mode="fortran"] alpha0 not None,
		np.ndarray[double,ndim=2,mode="c"] G not None,
		np.ndarray[double,ndim=2,mode="c"] result not None,
		double e0,
		double err=1e-6,
		int k=0):
	batch_omp_wrapper(<double*>np.PyArray_DATA(alpha0),<double*>np.PyArray_DATA(G),<double*>np.PyArray_DATA(result),alpha0.shape[0],G.shape[0],G.shape[1],e0,err,k)
	
#将Cython中的numpy数组传递给需要动态分配数组的C函数：http://www.pythonheidong.com/blog/article/129343/，fortran表示列优先，c表示行优先，np.ndarray[double,ndim=2,mode="fortran"]用于接受一个numpy列向量
#cython入门示例：https://www.jianshu.com/p/9410db8fbf50