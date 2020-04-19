import numpy as np
'''
calc salience patch dist between probe and gallery
References:
[1] https://github.com/Robert0812/salience_reid
'''

def sladdrowcols(X,vrow,vcol):
    vrow=np.mat(vrow)
    vcol=np.mat(vcol).T
    X=X+vrow
    X=X+vcol
    return X

def slmetric_pw(X1,X2): #eucdist
    sqs1=np.sum(X1*X1,axis=0)
    sqs2=np.sum(X2*X2,axis=0)
    M=(-2)*np.dot(X1.T,X2)
    M=sladdrowcols(M,sqs2,sqs1)
    M=np.maximum(M,0)
    return np.sqrt(M)

def mutualmap(feat1,feat2,nx,ny):
    '''apply patch matching, compute the pairwise distances within horizontal stripes. 
       feat1 or feat2 has shape(d,nxny)'''
    distmat=np.zeros((ny,nx))
    for i in range(ny):
        index=np.arange(i*nx,(i+1)*nx) #note that the extracted patches is arranged 
                     #with C style(not Fortran style in [1]) in extractDescriptors.py
        pv1=feat1[:,index]
        pv2=feat2[:,index]
        pwdist=slmetric_pw(pv1,pv2)
        distmat[i,:]=np.min(np.asarray(pwdist),axis=1)
    return distmat

def phiFun(x, m1, sigmal,nor):
    # phi{i} = phiFun(D_cell{i}, mp_cell{i}, para);
    # phiFun = @(x, m1, par) (exp(-double(x).^2/par.sigma1^2).*m1).^par.nor;
    resultMatrix=(np.exp((-x**2)/(sigmal**2))*m1)**nor
    sumResult=np.sum(np.reshape(resultMatrix,(resultMatrix.size,)))
    return sumResult

def calc_patch_pwdist(probe,gallery,nx,ny):
    '''return distance matrix of probe and gallery. nx is the num of patches in x-axis and 
       ny is the num of patches in y-axis and '''
    if probe.shape!=gallery.shape:
        raise ValueError('(single-shot)probe\'s shape must be equal to gallery\'s shape!')
    d,nxny,n=probe.shape #d is the dim of one patch, n is the imgs num in 
                         #probe or gallery, nxny is total patch num in one image
    distanceMatrices=[]
    for i in range(n):
        if i%10==0:
            print('now processing(salience matching) of sample %d'%i,end='\r')
        prbi=probe[...,i]
        for j in range(n):
            galj=gallery[...,j]
            distanceMatrices.append(mutualmap(prbi,galj,nx,ny))
    phiResults = []
    sigmal=2.8
    nor=2
    for i in range(len(distanceMatrices)):
        phiResult = phiFun(distanceMatrices[i],np.ones((ny, nx), dtype=np.int),sigmal, nor)
        phiResults.append(phiResult)
    patchLevelDist = np.mat(phiResults).reshape(n, -1).T
    return patchLevelDist