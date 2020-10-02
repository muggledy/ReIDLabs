'''
多粒度网络（MGN）
重排后结果（参数：lr=0002,num_epochs=80,step_size=60,gamma=0.1，另外所有损失
权重系数全部置为1）：
Rank-1:94.45% Rank-5:97.27% Rank-10:97.86% Rank-20:98.75% Rank-100:99.70%
mAP:92.35
用时1.62小时
设置num_epochs=100，结果基本不变：
Rank-1:94.89% Rank-5:97.62% Rank-10:98.16% Rank-20:98.93% Rank-100:99.67%
mAP:92.79
如果不做重排，结果为：
Rank-1:93.68% Rank-5:97.89% Rank-10:98.66% Rank-20:99.29% Rank-100:99.67%
mAP:82.98
注：结果之所以和论文有所不同，是因为我没有使用它的训练参数
'''

from initial import *
from deep.data_manager import Market1501
from deep.sampler import RandomIdSampler
from deep.data_loader import load_dataset, \
    default_train_transforms,default_test_transforms
from deep.models.ResNet import ResNet50_MGN
from deep.loss import TripletHardLoss
from deep.train import train,setup_seed
from deep.test import test,extract_feats
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint
from deep.plot_match import plot_match
import torchvision.transforms as T
import torch as pt
import torch.nn as nn
from functools import partial
from zoo.plot import plot_dataset
from deep.loss import CrossEntropyLabelSmooth

if __name__=='__main__':
    setup_seed(0)
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/Market-1501-v15.09.15/')
    checkpoint=CheckPoint()
    checkpoint.load('ResNet50_MGN.tar')

    sampler=partial(RandomIdSampler,num_instances=4)
    market1501=Market1501(dataset_dir)
    market1501.print_info()
    default_train_transforms[0]=T.Resize((384,128))
    default_test_transforms[0]=T.Resize((384,128))
    train_iter,query_iter,gallery_iter=load_dataset(market1501,32,32,sampler=sampler, \
        train_transforms=default_train_transforms,test_transforms=default_test_transforms)

    num_classes=len(set(list(zip(*market1501.trainSet))[1]))
    net=ResNet50_MGN(num_classes)

    softmax_loss=nn.CrossEntropyLoss()
    # softmax_loss=CrossEntropyLabelSmooth(num_classes) #LSR Loss，没有带来提升，反而降低了0.2%
    triplet_loss=TripletHardLoss(margin=0.3)

    losses=(triplet_loss,triplet_loss,triplet_loss,softmax_loss,softmax_loss,softmax_loss, \
        softmax_loss,softmax_loss,softmax_loss,softmax_loss,softmax_loss)

    lr,num_epochs=0.0002,100
    optimizer=pt.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.StepLR(optimizer,step_size=60,gamma=0.1)

    train(net,train_iter,losses,optimizer,num_epochs,scheduler,checkpoint=checkpoint, \
        coeffis=None,use_amp=False) #coeffis设为None，表示所有子损失融合权重都为1

    save_gal_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/market1501_resnetMGN_gallery.mat')
    re_calc_gal_fea=True #如果（每当）模型重新训练了，要记得置为True，之后可以再改回False节省时间
    test(net,query_iter,save_gal_path if not re_calc_gal_fea and os.path.exists(save_gal_path) \
        else gallery_iter,eval_cmc_map,save_galFea=save_gal_path,re_rank=False)

    plot_samples=True
    persons=9 #可视化测试集样本，需要的参数除了样本特征，还有其ID以及所在摄像头编号，既然三元组损失
              #或者其他损失的目的都是使相同类别行人距离相近、使不同类行人距离相远，那么在测试集上可以
              #展示出聚类的效果，注意不同类行人示以不同的颜色，不同摄像头示以不同的记号，考虑到颜色匮
              #乏，所以展示的行人数量不能太多，必须加以控制
    dim=3
    if plot_samples:
        query_feats=extract_feats(net,query_iter)
        gallery_feats=extract_feats(net,gallery_iter)
        test_feats=np.concatenate((query_feats,gallery_feats),axis=1)
        query_pids,query_cids=list(zip(*(market1501.querySet)))[1:]
        gallery_pids,gallery_cids=list(zip(*(market1501.gallerySet)))[1:]
        test_pids=query_pids+gallery_pids
        test_cids=query_cids+gallery_cids

        unique_k_pids=np.random.permutation(list(set(test_pids)))[:persons]
        inds=np.where(np.sum(np.array(unique_k_pids)[:,None]==test_pids,axis=0)==1)[0]
        
        plot_dataset(test_feats[:,inds],np.array(test_pids)[inds],np.array(test_cids)[inds],dim=dim)

    plot_match(net,query_gallery=market1501,galfeas_path=save_gal_path)