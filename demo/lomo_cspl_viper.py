'''
I can't reproduct the result(42.97%) of paper(CSPL), only 29%!
I don't know where the problem is?
'''

from initial import *
from zoo.viper import get_lomo_viper
from zoo.cspl import cspl
from sklearn.decomposition import PCA

cwd=getcwd(__file__)
numClass=632
numFlods=10

t1=time.time()
# --Stage1: get lomo features--
probFea,galFea=get_lomo_viper() #mask='ellipse' no help, see note 2

'''use official lomo features(with retinex), compared to mine, no difference
import scipy.io as sio
data=sio.loadmat(os.path.normpath( \
    os.path.join(cwd,'../data/viper_lomo.mat')))['descriptors']
probFea=data[:,:632].T
galFea=data[:,632:].T
'''

probFea=probFea.T
galFea=galFea.T
# --Stage2: train and match--
dim=100 #for viper, set to 600(99%) and set to 100(56%), no difference!
        #PCA refer: https://www.cnblogs.com/pinard/p/6243025.html
cs=[]
for i in range(numFlods):
    p=np.random.permutation(numClass)
    half=int(numClass/2)
    probFea1=probFea[:,p[:half]]
    galFea1=galFea[:,p[:half]]

    pca=PCA(n_components=dim)
    t=pca.fit_transform(np.hstack((probFea1,galFea1)).T)
    probFea1,galFea1=np.split(t,2)
    energy=sum(pca.explained_variance_ratio_)
    print('reduct to dim %d(energy:%.2f%%)'%(dim,energy*100))

    P1,P2,A=cspl(probFea1.T,galFea1.T,iter=200)
    probFea2=probFea[:,p[half:]]
    galFea2=galFea[:,p[half:]]
    t=pca.transform(np.hstack((probFea2,galFea2)).T)
    probFea2,galFea2=np.split(t,2)

    dist=euc_dist(P1.dot(probFea2.T),A.dot(P2.dot(galFea2.T)))
    c=calc_cmc(dist.T,range(half),range(half),rank=100)
    cs.append(c)
cs=np.array(cs)
c_mean=np.mean(cs,axis=0)
print('all time consumes:',time.time()-t1)
plot_cmc(c_mean,labels=['viper'],verbose=True)