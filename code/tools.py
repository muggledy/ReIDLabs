import numpy as np

def euc_dist(X,Y):
    '''计算数据集X和Y中各个样本之间欧式距
    离²，返回矩阵列方向代表X，行方向代表Y'''
    A=np.sum(X.T*(X.T),axis=1)
    D=np.sum(Y.T*(Y.T),axis=1)
    return A[:,None]+D[None,:]-X.T.dot(Y)*2
