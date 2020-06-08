'''
This work is previous of udlgl by author Elyor Kodirov, we use the same data as 
./demo_laplacianl1_kcca_viper.py. Because of the complexity of it's optimization part(see paper 
"Graph Regularized Sparse Coding for Image Representation"), i.e. feature sign search,  so i 
modify the objective function(||Y||_1=>||Y||_F) accroding to my boss's guidance
Issue:
Paper[1] claims that rank-1 of viper can achieve 30%(cam_b as probe), but in fact there is a 
flaw in the public source code, only has 23.x% if you shuffle the training data firstly(in matlab), 
the problem lies in the construction of the affinity matrix W(see graph_cross_view.m in source code)
References:
[1] Kodirov, Elyor & Xiang, Tao & Gong, Shaogang. (2015). Dictionary Learning with Iterative 
    Laplacian Regularisation for Unsupervised Person Re-identification. 44.1-44.12. 10.5244/C.29.44. 
'''

import numpy as np
from demo_laplacianl1_kcca_viper import get_data
from code.udlgl import get_cross_view_graph
from code.optimize import opt_coding_l2
from code.lomo.tools import calc_cmc,plot_cmc
from code.tools import cosine_dist,print_cmc,norm_labels_simultaneously
from code.dllap import dllap

k_nn=3
nBasis=2**8
alpha=1.0
beta=0.0001
nIters=50
lambd1=0.04 #this parametr must be tuned

cs=[]
D=None
for trial in range(1):
    print('Trial-%d...'%(trial+1))
    data=get_data(trial)
    probFea1,galFea1=data['trainA_feats'],data['trainB_feats']
    # probFea1,galFea1=probFea1[:,np.random.permutation(probFea1.shape[1])], \
    #     galFea1[:,np.random.permutation(galFea1.shape[1])] #verify: shuffle samples' order won't 
                                                           #affect result if don't use backdoor 
                                                           #in func get_cross_view_graph, this result 
                                                           #is reliable
    probFea2,galFea2=data['testA_feats'],data['testB_feats']
    
    for in_iter in range(1):
        print('1.construct graph(W) similarity matrix...')
        if in_iter==0:
            W_full=get_cross_view_graph(probFea1,galFea1,k=k_nn,backdoor=True)
        elif in_iter==1:
            print('use the learned samples\' coding this time')
            W_full=get_cross_view_graph(train_a_after_D,train_b_after_D,k=k_nn,backdoor=True)

        print('2.learn dictionary(D) with L2 laplacian...')
        D=dllap(np.hstack((probFea1,galFea1)), W_full, nBasis, alpha, beta, nIters, Dinit=D)

        train_a_after_D,P=opt_coding_l2(D,probFea1,lambd1)
        train_b_after_D=P.dot(galFea1)

    test_a_after_D,P=opt_coding_l2(D,probFea2,lambd1)
    test_b_after_D=P.dot(galFea2)
    dist=cosine_dist(test_a_after_D,test_b_after_D)
    print('calc CMC...')
    c=calc_cmc(dist,*norm_labels_simultaneously(data['testB_labels'][0],data['testA_labels'][0]),100) #
                                                       #note here: cam_b as probe, if cam_a as probe, 
                                                       #rank scores will decrease a little
    cs.append(c)
    print_cmc(c)

print('average CMC:')
c_mean=np.mean(np.array(cs),axis=0)
print_cmc(c_mean,color=True)
plot_cmc(c_mean,['viper'],verbose=True)