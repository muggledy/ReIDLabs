'''
References:
[1] H.-X. Yu, A. Wu and W.-S. Zheng, "Cross-view Asymmetric Metric Learning for Unsupervised 
    Person Re-identification", In ICCV, 2017.
muggledy 2020/5/14
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from functools import reduce
from itertools import count
from numpy.linalg import inv,eig
from .tools import norm_labels,construct_I
from .cprint import cprint_err,cprint_out

class X_data:
    def initialize_data(self,data,id_views):
        '''get resorted data [x_1^1,x_2^1,...,x_n1^1,...,x_1^V,x_2^V,...,x_nV^V] and 
           corresponding id_views [1,1,...,1,...,V,V,...,V], we also calc split 
           boundary [1,n1,n1+n2,...,n1+n2+...+nV] of different cameras for convenience'''
        d,n=data.shape
        if n!=len(id_views):
            raise ValueError('sample num must be equal to num of id_views!')
        id_views=norm_labels(id_views) #note here
        views=np.unique(id_views)
        self.data=np.zeros((d,n))
        self.id_views=np.zeros((n,))
        self.originPos=np.zeros((n,)) #unused
        start=0
        for i in views:
            inds=np.where(id_views==i)[0]
            self.originPos[start:start+len(inds)]=inds
            self.data[:,start:start+len(inds)]=data[:,inds]
            self.id_views[start:start+len(inds)]=i
            start+=len(inds)
        id_views_groups=reduce(lambda x,y:x+[[y]] if x[-1][-1]!=y else x[:-1]+[x[-1]+[y]], \
                                self.id_views.astype('int'),[['^']])[1:]
        self.boundary=np.cumsum([0]+[len(i) for i in id_views_groups])
        
    def __getitem__(self,key):
        '''key should be an integer that represents the num(i) of camera, and we will 
           return data(d,ni) in this camera view'''
        return self.data[:,self.boundary[key]:self.boundary[key+1]]

    @property
    def info(self):
        '''return info abount X_data obj, including the num of cameras, sample num of 
           each camera, dim of sample and num of all samples'''
        info_dict={}
        info_dict['num_cameras']=len(self.boundary)-1
        info_dict['num_view_samples']=self.boundary[1:]-self.boundary[:-1]
        info_dict['dim_sample']=self.data.shape[0]
        info_dict['num_samples']=self.data.shape[1]
        return info_dict

def construct_X_(X_data_obj):
    '''return X_=diag{X1,...,XV}, where Xi means these samples captured in i-th camera view'''
    d,N=X_data_obj.data.shape
    V=X_data_obj.info['num_cameras']
    X_=np.zeros((d*V,N))
    for i in range(V):
        X_[i*d:(i+1)*d,X_data_obj.boundary[i]:X_data_obj.boundary[i+1]]=X_data_obj[i]
    return X_

def construct_H(id_identities):
    '''return H(N,k) which each column represents an indicator vector, k is the num of unique 
       elems in id_identities, i.e. the number of identity classes, N is the num of all samples'''
    # N=len(id_identities)
    # ids=np.unique(id_identities)
    # H=np.zeros((N,len(ids)))
    # for i in ids:
    #     inds=np.where(id_identities==i)[0]
    #     H[inds,i]=1/np.sqrt(len(inds))
    # return H
    N=len(id_identities)
    k=len(np.unique(id_identities))
    H=(np.arange(k)[None,:]==id_identities[:,None]).astype('float32')
    return H/np.sqrt(np.sum(H,axis=0))
    
def construct_cov_(X_data_obj):
    '''return cov_=diag{cov1,...,covV}, where covi is the covariance matrix of data in camera i'''
    V=X_data_obj.info['num_cameras']
    d,_=X_data_obj.data.shape
    cov_=np.zeros((d*V,d*V))
    for i in range(V):
        Xi=X_data_obj[i]
        cov_[i*d:(i+1)*d,i*d:(i+1)*d]=Xi.dot(Xi.T)/(Xi.shape[1])
    alpha=1
    cov_=cov_+alpha*np.trace(cov_)/(d*V)*np.eye(d*V) #here the regularization is 
                                                     #very very important, if you 
                                                     #just add 0.001, you will 
                                                     #get a bad cmc rank, bad!
    return cov_

def construct_D(X_data_obj):
    '''return D = [(V-1)E -E -E ... -E
                   -E (V-1)E -E ... -E
                          . . .
                   -E -E ...    (V-1)E] 
       with size (V*d,V*d)'''
    V=X_data_obj.info['num_cameras']
    d,_=X_data_obj.data.shape
    I=construct_I((V,V),d)
    return np.eye(V*d)*V-I

def construct_U_(eigV,eigU,nBasis,cov_,nViews):
    '''select some basis(u) from eigU that s.t. UᵀMU=VI => uᵀMu=V, M is the cov_. If uᵀMu=p, 
       both sides multiply by V/p, where V is the nViews'''
    inds=np.argsort(eigV)[:nBasis] #select the smallest eigens to minimize the obj
    us_unnormalized=eigU[:,inds]
    U_=np.zeros(us_unnormalized.shape)
    for i in range(U_.shape[1]):
        u_un=us_unnormalized[:,[i]]
        u=u_un*np.sqrt(nViews)/np.sqrt(u_un.T.dot(cov_).dot(u_un))
        U_[:,[i]]=u
    return U_

def camel(data,id_views,k,lambd,nBasis=None,eps=10**(-8),kmeans_max_iter=100,max_iter=20):
    '''the train data(d,N) contains samples from several cameras, id_views(N,) indicates the 
       camera-view id of each sample. CAMEL learns a projection subpace in which the 
       same pedestrian images from different views will be close, func will return a list of 
       asymmetric project matrixes of all camera view'''
    X=X_data()
    X.initialize_data(data,id_views)
    X_=construct_X_(X)
    kmeans=KMeans(n_clusters=k,max_iter=kmeans_max_iter)
    id_identities=kmeans.fit_predict(X.data.T)
    H=construct_H(id_identities)
    cov_=construct_cov_(X)
    D=construct_D(X)
    d,N=X.data.shape
    X_X_T=X_.dot(X_.T)
    inv_cov_=inv(cov_)
    G=inv_cov_.dot(lambd*D+X_X_T/N-X_.dot(H).dot(H.T).dot(X_.T)/N)
    eigV,eigU=eig(G)
    nBasis=d if nBasis==None else nBasis
    V=X.info['num_cameras']
    U_=construct_U_(eigV,eigU,nBasis,cov_,V)
    calc_Fobj=lambda X_,U_,H,D,X_X_T,N,lambd:lambd*np.trace(U_.T.dot(D).dot(U_))+ \
                (np.trace(U_.dot(U_.T).dot(X_X_T))- \
                np.trace(H.T.dot(X_.T).dot(U_).dot(U_.T).dot(X_).dot(H)))/N
    Fobj=calc_Fobj(X_,U_,H,D,X_X_T,N,lambd)
    old_Fobj=Fobj
    for i in count():
        if i>=max_iter:
            cprint_err('(CAMEL)max iter(%d)!'%max_iter)
            break
        else:
            print('iter:%d'%i,end='\r')
        Y=U_.T.dot(X_) #(d',N), d' is the dim of sample in project space
        id_identities=kmeans.fit_predict(Y.T)
        H=construct_H(id_identities) #do kmeans to update H to update U_
        G=inv_cov_.dot(lambd*D+X_X_T/N-X_.dot(H).dot(H.T).dot(X_.T)/N)
        U_=construct_U_(*eig(G),nBasis,cov_,V)
        Fobj=calc_Fobj(X_,U_,H,D,X_X_T,N,lambd)
        if Fobj-old_Fobj>0:
            cprint_out('(CAMEL)convergence(%d)!'%(i+1))
            break
        else:
            old_Fobj=Fobj
    return np.split(U_,V,axis=0)

def proj_data(data,id_views,Us):
    '''proj data into subspace, Us is the list of all project matrix, id_views indicate which 
       camera the data comes from,note that id_views must be in range[0,V-1]!'''
    views=np.unique(id_views)
    if np.max(views)>=len(Us) or np.min(views)<0:
        raise ValueError('id_views\' value must be in range[0,%d]!'%(len(Us)-1))
    _,n=data.shape
    _,d_=Us[0].shape
    ret=np.zeros((d_,n)) #(d',n)
    for i in range(n):
        sam=data[:,[i]]
        ret[:,[i]]=Us[id_views[i]].T.dot(sam)
    return ret

if __name__=='__main__':
    from sklearn.datasets import make_blobs
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    #demo of k-means by sklearn
    centers=4
    X, y_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=4) #https://juejin.im/post/5daffd26e51d45249f6085a5
    y_kmeans = kmeans.fit_predict(X)
    colors=cm.gist_rainbow(Normalize(0,1)(np.linspace(0,1,centers)))
    plt.scatter(X[:,0], X[:, 1],c=colors[y_kmeans]) #or plt.scatter(X[:,0], X[:, 1],c=y_kmeans,cmap='gist_rainbow')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:, 1], c='black', s=80, marker='x')
    plt.show()