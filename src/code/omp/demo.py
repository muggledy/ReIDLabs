#测试样例，在python中调用动态链接库（c++代码编译结果）

from batch_omp import omp
import numpy as np
from timeit import repeat

D=np.array([[-0.707,0.8,0],[0.707,0.6,-1]])
x=np.array([1.65,-0.25]).reshape(-1,1)

result=np.empty((D.shape[1],1)) #用于存放结果

omp(D.T.dot(x),D.T.dot(D),result,x.reshape(-1).dot(x)[0])
print(result)

print(np.mean(repeat('omp(D.T.dot(x),D.T.dot(D),result,x.reshape(-1).dot(x)[0])','from batch_omp import omp;from __main__ import D,x,result',number=1000))) #相比于之前的numpy版本（0.16455916666666667），效率提升了十倍（当前：0.015921466666666665）