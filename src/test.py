import numpy as np
from code.udlgl import get_cross_view_graph
from demo_laplacianl1_kcca_viper import get_data
from code.pdllap import pdllap
from code.optimize import opt_coding_l2
from code.tools import cosine_dist,print_cmc,norm_labels_simultaneously
from code.lomo.tools import calc_cmc,plot_cmc
from sklearn.decomposition import PCA

k_nn=3
nBasis=800 #number of dictionary atoms
nIters=4 #iteration number
ndim=630 #dim after project
lambd1=1 #proj matrix(not 0) constraint
lambd2=.0001 #sparsity constraint
lambd3=1 #graph constraint
lambd4=.0001 #affinity matrix constraint
lambd0=.01

cs=[]
for trial in range(1):
    print('Trial-%d...'%(trial+1))
    data=get_data(trial)
    probFea1,galFea1=data['trainA_feats'],data['trainB_feats']
    probFea2,galFea2=data['testA_feats'],data['testB_feats']

    #PCA
    pca=PCA(n_components=630)
    t=pca.fit_transform(np.hstack((probFea1,galFea1)).T)
    probFea1,galFea1=np.split(t.T,2,axis=1)
    probFea2=pca.transform(probFea2.T).T
    galFea2=pca.transform(galFea2.T).T

    # probFea1,galFea1=probFea1[:,np.random.permutation(probFea1.shape[1])], \
    #     galFea1[:,np.random.permutation(galFea1.shape[1])]
    
    for in_iter in range(1):
        print('1.construct graph(W) similarity matrix...')
        if in_iter==0:
            W_full=get_cross_view_graph(probFea1,galFea1,k=k_nn,backdoor=True)
        elif in_iter==1:
            print('use the learned samples\' coding this time')
            W_full=get_cross_view_graph(train_a_after_D,train_b_after_D,k=k_nn,backdoor=True)

        print('2.learn dictionary and proj martix by PDLLAP...')
        D,Us=pdllap(np.hstack((probFea1,galFea1)),[0]*probFea1.shape[1]+[1]*galFea1.shape[1] \
            ,W_full,k_nn,nBasis,ndim,lambd1,lambd2,lambd3,lambd4,nIters,use_cov=True)

        train_a_after_D,P=opt_coding_l2(D,Us[0].T@probFea1,lambd0)
        train_b_after_D=P.dot(Us[1].T@galFea1)

    test_a_after_D,P=opt_coding_l2(D,Us[0].T@probFea2,lambd0)
    test_b_after_D=P.dot(Us[1].T@galFea2)
    dist=cosine_dist(test_a_after_D,test_b_after_D)
    print('calc CMC...')
    c=calc_cmc(dist,*norm_labels_simultaneously(data['testB_labels'][0], \
        data['testA_labels'][0]),100) #cam_b as probe
    cs.append(c)
    print_cmc(c)

print('average CMC:')
c_mean=np.mean(np.array(cs),axis=0)
print_cmc(c_mean,color=True)
plot_cmc(c_mean,['viper'],verbose=True)