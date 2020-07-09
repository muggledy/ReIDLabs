'''
I can't implement the kernel version of olda correctly! Help...
'''

from initial import *
from zoo.viper import get_lomo_viper,get_gog_viper
from zoo.olda import olda,kolda

t1=time.time()

# --Stage1: get lomo features--
probFea,galFea=get_lomo_viper() #35%
#probFea,galFea=get_gog_viper() #use GOG features: 43%
probFea=probFea.T
galFea=galFea.T

'''official lomo
import scipy.io as sio
import os.path
cwd=os.path.dirname(__file__)
data=sio.loadmat(os.path.normpath(os.path.join(cwd,'../data/viper_lomo.mat')))['descriptors']
probFea=data[:,:632]
galFea=data[:,632:]
'''

# --Stage2: training and matching--
numClass=632
numFlods=1
cs=[]
for i in range(numFlods):
    p=np.random.permutation(numClass)
    half=int(numClass/2)
    probFea1=probFea[:,p[:half]]
    galFea1=galFea[:,p[:half]]
    L=olda(np.hstack((probFea1,galFea1)),p[:half].tolist()*2)
    probFea2=probFea[:,p[half:]]
    galFea2=galFea[:,p[half:]]
    dist=euc_dist(L.T.dot(probFea2),L.T.dot(galFea2))
    c=calc_cmc(dist.T,range(half),range(half),rank=100)
    cs.append(c)
cs=np.array(cs)
c_mean=np.mean(cs,axis=0)
print('all time consumes:',time.time()-t1)
plot_cmc(c_mean,labels=['viper'],verbose=True)

### visualize in 2-D space
from zoo.plot import plot_dataset
plot_dataset(np.hstack((L.T.dot(probFea1),L.T.dot(galFea1))),list(range(half))*2,[0]*half+[1]*half,'PCA',3) #you can see almost all train data points convergence
plot_dataset(np.hstack((L.T.dot(probFea2),L.T.dot(galFea2))),list(range(half))*2,[0]*half+[1]*half,'PCA',3) #see test data points
###

# --kernel version(now has error!!!)--
# numClass=632
# numFlods=1
# cs=[]
# for i in range(numFlods):
#     p=np.random.permutation(numClass)
#     half=int(numClass/2)
#     probFea1=probFea[:,p[:half]]
#     galFea1=galFea[:,p[:half]]
#     L,K=kolda(np.hstack((probFea1,galFea1)),p[:half].tolist()*2,kernel='gaussian',sigma=1)
#     probFea2=probFea[:,p[half:]]
#     galFea2=galFea[:,p[half:]]
#     dist=euc_dist(L.T.dot(K(probFea2)),L.T.dot(K(galFea2)))
#     c=calc_cmc(dist.T,range(half),range(half),rank=100)
#     cs.append(c)
# cs=np.array(cs)
# c_mean=np.mean(cs,axis=0)
# print('all time consumes:',time.time()-t1)
# plot_cmc(c_mean,labels=['viper'],verbose=True)