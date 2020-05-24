'''
this work is previous of udlgl by author Elyor Kodirov, download data(VIPeR_data_trial_1.mat) 
from https://sites.google.com/site/elyorkodirovresearch/publications into ../data/. Because of 
the complexity of it's optimization part(see paper "Graph Regularized Sparse Coding for Image 
Representation"), so i use the matlab engine temporarily to run source matlab code
(./code/test/py_matlab_udlgl2), and get 31.65%(cam_a as gallery, if cam_a as probe, only 25.31%)
References:
[1] Kodirov, Elyor & Xiang, Tao & Gong, Shaogang. (2015). Dictionary Learning with Iterative 
    Laplacian Regularisation for Unsupervised Person Re-identification. 44.1-44.12. 10.5244/C.29.44. 
'''

import scipy.io as scio
import numpy as np
import os
from code.udlgl import get_cross_view_graph
from code.optimize import opt_coding_l2
from code.lomo.tools import calc_cmc,plot_cmc
from code.tools import cosine_dist
from code.graph_sc import graph_sc

cur_dir=os.path.dirname(__file__)
data=scio.loadmat(os.path.join(cur_dir,'../data/VIPeR_data_trial_1.mat'))

k_nn=3
nBasis=2**8
alpha=1.0
beta=0.0001
nIters=50

print('Stage1: Train')
for out_iter in range(2):
    print('Iter: %d'%out_iter)
    if out_iter==0:
        X1=data['Cam_A_tr']
        X2=data['Cam_B_tr']
    elif out_iter==1:
        print('use the learned samples\' coding this time')
        X1=train_a_after_D.T
        X2=train_b_after_D.T
    
    print('1. construct graph(W) similarity matrix ...')
    W_full=get_cross_view_graph(X1.T,X2.T,k=k_nn,backdoor=True)

    print('2. learn dictionary(D) with L1 laplacian ...')
    D,*_=graph_sc(np.vstack((data['Cam_A_tr'],data['Cam_B_tr'])).T, W_full, nBasis, alpha, beta, nIters)

    lambd1=0.04 #this parametr must be tuned
    train_a_after_D=opt_coding_l2(D,data['Cam_A_tr'].T,lambd1)
    train_b_after_D=opt_coding_l2(D,data['Cam_B_tr'].T,lambd1)
    # break
print('Stage2: Test')
test_a_after_D=opt_coding_l2(D,data['Cam_A_te'],lambd1)
test_b_after_D=opt_coding_l2(D,data['Cam_B_te'],lambd1)

dist=cosine_dist(test_a_after_D,test_b_after_D)
print('calc CMC ...')
c=calc_cmc(dist.T,np.arange(316),np.arange(316),100)
print(c)
#print(calc_cmc(dist,np.arange(316),np.arange(316),100)[[0,4,9,19]])
plot_cmc(c,['viper'],verbose=True)

'''CALL MATLAB
import scipy.io as scio
import numpy as np
import os
from code.udlgl import get_cross_view_graph
from code.optimize import opt_coding_l2
from code.lomo.tools import calc_cmc,plot_cmc
from code.tools import cosine_dist
from importlib import reload

cur_dir=os.path.dirname(__file__)
data=scio.loadmat(os.path.join(cur_dir,'../data/VIPeR_data_trial_1.mat'))

k_nn=3
nBasis=2**8
alpha=1.0 #must be float! otherwise when you call matlab program, you will get error!
beta=0.0001
nIters=50

temp_var_file=os.path.join(cur_dir,'../data/graphsc_para.mat')
temp_origin_work_path=cur_dir
temp_graphsc_work_path=os.path.join(cur_dir,'./code/test/py_matlab_udlgl2/')
temp_D_file=os.path.join(cur_dir,'../data/graphsc_result.mat')

print('Stage1: Train')
for out_iter in range(2):
    print('Iter: %d'%out_iter)
    if out_iter==0:
        X1=data['Cam_A_tr']
        X2=data['Cam_B_tr']
    elif out_iter==1:
        print('use the learned samples\' coding this time')
        X1=train_a_after_D.T
        X2=train_b_after_D.T
    
    print('1. construct graph(W) similarity matrix ...')
    W_full=get_cross_view_graph(X1.T,X2.T,k=k_nn,backdoor=True)

    print('2. learn dictionary(D) with L1 laplacian ...')
    scio.savemat(temp_var_file,{'X_tr':np.vstack((data['Cam_A_tr'],data['Cam_B_tr'])).T, \
        'nBasis':nBasis,'alpha':alpha,'beta':beta,'nIters':nIters,'W_full':W_full})
    
    os.chdir(temp_graphsc_work_path)
    print('switch work path to %s'%temp_graphsc_work_path)
    if out_iter==0:
        import code.test.py_matlab_udlgl2.test_demo #call matlab engine
    elif out_iter==1:
        reload(code.test.py_matlab_udlgl2.test_demo)
    os.chdir(temp_origin_work_path)
    print('switch to origin work path %s'%temp_origin_work_path)

    D=scio.loadmat(temp_D_file)['D']
    lambd1=0.04 #this parametr must be tuned
    train_a_after_D=opt_coding_l2(D,data['Cam_A_tr'].T,lambd1)
    train_b_after_D=opt_coding_l2(D,data['Cam_B_tr'].T,lambd1)
    # break
print('Stage2: Test')
test_a_after_D=opt_coding_l2(D,data['Cam_A_te'],lambd1)
test_b_after_D=opt_coding_l2(D,data['Cam_B_te'],lambd1)

dist=cosine_dist(test_a_after_D,test_b_after_D)
print('calc CMC ...')
c=calc_cmc(dist.T,np.arange(316),np.arange(316),100)
print(c) #25.31%
#print(calc_cmc(dist,np.arange(316),np.arange(316),100)[[0,4,9,19]]) #31.65%
plot_cmc(c,['viper'],verbose=True)
'''