'''
References:
[1] Belhumeur P N , João P. Hespanha, Kriegman D J . Eigenfaces vs. Fisherfaces: 
    Recognition Using Class Specific Linear Projection[M]// Computer Vision — ECCV 
    '96. IEEE Computer Society, 2006.
muggledy 2020/4/3
'''

import numpy as np
from sklearn.decomposition import PCA
#from .lomo.tools import measure_time
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../'))
from zoo.tools import norm_labels
from functools import reduce

def get_Hw_Hb_Ht(X,labels):
    '''refer to olda's reference paper[1]'''
    labels=norm_labels(labels)
    sortedInd=np.argsort(labels)
    labels=labels[sortedInd]
    X=X[:,sortedInd]
    groups=reduce(lambda x,y:x+[[y]] if x[-1][-1]!=y else x[:-1]+[x[-1]+[y]],labels,[['^']])[1:]
    lenGroups=np.array([len(i) for i in groups])
    rightEdge=np.cumsum(lenGroups)
    leftEdge=rightEdge-lenGroups
    rightEdge-=1
    c=len(groups)
    d,n=X.shape
    Hw=np.zeros((d,n))
    Hb=np.zeros((d,c))
    for i in range(c):
        l,r=leftEdge[i],rightEdge[i]+1
        t=X[:,l:r].dot(np.ones((r-l,1)))*(1/(r-l))
        Hb[:,[i]]=t
        Hw[:,l:r]=X[:,l:r]-t
    coef=1/np.sqrt(n)
    Hw*=coef
    center=X.dot(np.ones((n,1)))*(1/n)
    Hb-=(center)
    Hb=Hb*np.sqrt(lenGroups)*coef
    Ht=(X-center)*coef
    return Hw,Hb,Ht

def lda(X,labels,n_components=None,regularizer=None):
    '''calc within-class matrix Sw and between-class matrix Sb, then do eigen 
       decomposition of (Sw)^-1(Sb), select top c-1 maximum eigen values' corresponding 
       eigen vectors as the linear project matrix, c is the num of identity-class. 
       As usual, n_components must be less than c-1, regularizer can be 0.01'''
    Hw,Hb,_=get_Hw_Hb_Ht(X,labels)
    c=len(np.unique(labels))
    Sw=Hw.dot(Hw.T)
    Sb=Hb.dot(Hb.T)
    eigVals,eigVects=np.linalg.eig(np.linalg.inv(Sw+(np.eye(X.shape[0])*regularizer if regularizer!=None else 0)).dot(Sb))
    maxSortedInd=np.argsort(eigVals)[::-1][:c-1 if n_components==None else n_components]
    return eigVects[:,maxSortedInd]

#@measure_time
def pca_lda(X,labels):
    '''X(d*n) is the training samples, each column represents one sample. X can be 
       unordered but you must provide their labels(id). We all know that LDA confronts 
       with SSS problem, and the within-class matrix will always be singular, paper[1] 
       proposed using PCA to reduce the dimension of the feature space to n-c firstly, 
       c is the num of pedestrians' classes, then doing LDA. Func will return the 
       project matrix P, given a sample set X, please use PᵀX'''
    d,n=X.shape
    X=np.float32(X)
    c=len(np.unique(labels))
    #99% memory usage! But sklearn's PCA does very well!(cattle beer)
    #X0=X-X.dot(np.ones((n,1)))*(1/n)
    #cov=X0.dot(X0.T)*(1/n) #covariance matrix
    #eigVals,eigVects=numpy.linalg.eig(cov)
    #maxSortedInd=np.argsort(eigVals)[::-1][:n-c]
    #W=eigVects[maxSortedInd]
    #X=W.T.dot(X)
    pca=PCA(n_components=n-c)
    pca.fit(X.T)
    WT=pca.components_
    L=lda(WT.dot(X),labels)
    return WT.T.dot(L)

if __name__=='__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    iris=datasets.load_iris() #iris dataset contains 150 rows(sample number), 4 columns
                              #(feature dimension), belongs to three classes(0,1,2)
    data_train,data_test,target_train,target_test=train_test_split(iris.data,iris.target,test_size=0.5)
    
    L=lda(data_train.T,target_train,2) #project to 2D
    data_test_2D=L.T.dot(data_test.T)
    class0_data_test_2D,class1_data_test_2D,class2_data_test_2D=data_test_2D[:,np.where \
            (target_test==0)],data_test_2D[:,np.where(target_test==1)], \
            data_test_2D[:,np.where(target_test==2)]
    plt.subplot(121)
    plt.scatter(*class0_data_test_2D)
    plt.scatter(*class1_data_test_2D)
    plt.scatter(*class2_data_test_2D)
    plt.legend(['class 0','class 1','class 2'])
    plt.xlabel('my LDA')
    
    plt.subplot(122)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(data_train, target_train)
    data_test_2D=lda.transform(data_test)
    data_test_2D=data_test_2D.T
    
    class0_data_test_2D,class1_data_test_2D,class2_data_test_2D=data_test_2D[:,np.where \
            (target_test==0)],data_test_2D[:,np.where(target_test==1)], \
            data_test_2D[:,np.where(target_test==2)]
    plt.scatter(*class0_data_test_2D)
    plt.scatter(*class1_data_test_2D)
    plt.scatter(*class2_data_test_2D)
    plt.legend(['class 0','class 1','class 2'])
    plt.xlabel('sklearn\'s LDA')
    plt.show()
