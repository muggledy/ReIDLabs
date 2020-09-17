'''
2019/11/26
'''

import numpy as np
from numpy.linalg import svd,pinv,qr
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../'))
from lomo.tools import *

@measure_time
def xqda(probX,galX,probLabels,galLabels,lambd=0.001):
    '''Cross-view Quadratic Discriminant Analysis
       parameters:
          probX: features of probe images, size: (d,n)
          probLabels: class labels of probe images
          galX: features of gallery images, size: (d,m)
          galLabels: class labels of gallery images
          lambd: the regularizer, default: 0.001
       returns:
          W: the subspace projection matrix we have learned, size: (d,r), where r is the dimension of subspace
          M: (∑'ɪ)⁻¹-(∑'ᴇ)⁻¹, the learned symmetrical metrix kernel, size: (r,r)
          inCov: ∑'ɪ, covariance matrix of the intra-personal difference class in subspace, size: (r,r)
          exCov: ∑'ᴇ, covriance matrix of the extra-personal difference class in subspace, size: (r,r)
    '''
    d,numGals=galX.shape #m
    numProbs=probX.shape[1] #n
    
    if d>numProbs+numGals: #？
        flag=True
        W,X=qr(np.hstack((probX,galX)))
        probX=X[:,:numProbs]
        galX=X[:,numProbs:]
        d=X.shape[0]
    
    labels=np.unique(np.concatenate((galLabels,probLabels)))
    c=len(labels)
    probW=np.zeros(numProbs) #√m1,√m1,...,√mc |X~
    galW=np.zeros(numGals) #√n1,√n1,...,√nc |Z~
    probClassSum=np.zeros((d,c)) #S
    galClassSum=np.zeros((d,c)) #R
    ni=0
    for k in range(c):
        probIndex,=np.where(probLabels==labels[k])
        mk=len(probIndex)
        probClassSum[:,k]=np.sum(probX[:,probIndex],axis=1)
        galIndex,=np.where(galLabels==labels[k])
        nk=len(galIndex)
        galClassSum[:,k]=np.sum(galX[:,galIndex],axis=1)
        ni=ni+nk*mk
        probW[probIndex]=np.sqrt(nk)
        galW[galIndex]=np.sqrt(mk)
    probSum=np.sum(probClassSum,axis=1)[:,None] #s
    galSum=np.sum(galClassSum,axis=1)[:,None] #r
    probCov=probX.dot(probX.T) #XXᵀ
    galCov=galX.dot(galX.T) #ZZᵀ
    probX*=probW #X~
    galX*=galW #Z~
    inCov=probX.dot(probX.T)+galX.dot(galX.T)-probClassSum.dot(galClassSum.T)-galClassSum.dot(probClassSum.T) #∑ɪ
    exCov=numGals*probCov+numProbs*galCov-probSum.dot(galSum.T)-galSum.dot(probSum.T)-inCov #∑ᴇ
    ne=numGals*numProbs-ni
    inCov=inCov/ni
    exCov=exCov/ne
    inCov=inCov+lambd*np.eye(d)
    u,s,v=svd(pinv(inCov).dot(exCov))
    r=np.sum(s>1)
    sIndex=np.argsort(s)[::-1][:r]
    s=s[sIndex]
    u=u[:,sIndex]
    
    if locals().get('flag'):
        W=W.dot(u) #W
    else:
        W=u
        
    inCov=u.T.dot(inCov).dot(u) #∑'ɪ
    exCov=u.T.dot(exCov).dot(u) #∑'ᴇ
    M=pinv(inCov)-pinv(exCov)
    
    return W,M,inCov,exCov

if __name__=='__main__':
    pass
    