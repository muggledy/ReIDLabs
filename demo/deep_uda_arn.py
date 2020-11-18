'''
ARN, domain adaption from Market1501(source domain) to DukeMTMC(target domain), see paper[1]
 ↑
shit（我心态崩了，试问作者，您能复现您的论文吗？就这也CVPR？抑或您只是不愿意把正确无误的代码放出来？）

[1] Li, Yu-Jhe, et al. "Adaptation and re-identification network: An unsupervised deep transfer 
    learning approach to person re-identification." Proceedings of the IEEE Conference on 
    Computer Vision and Pattern Recognition Workshops. 2018.
'''

from initial import *
from deep.data_manager import Market1501,DukeMTMC
from deep.data_loader import load_dataset,load_train_iter, \
    load_query_or_gallery_iter,T
from deep.uda.ARN import arn
from deep.uda.ARN.loss import *
from deep.train import train,setup_seed
from deep.test import test
from deep.sampler import RandomIdSampler
from deep.eval_metric import eval_market1501,eval_cmc_map
from deep.models.utils import CheckPoint
import torch as pt
import torch.nn as nn
from deep.plot_match import plot_match
from functools import partial
pt.multiprocessing.set_sharing_strategy('file_system')

if __name__=='__main__':
    # pt.cuda.empty_cache() #no work
    setup_seed(0)
    checkpoint=CheckPoint()
    checkpoint.load('ARN_market.tar')
    train_transform_list = [T.RandomHorizontalFlip(p=0.5), T.Resize(size=(256, 128)),\
                        T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    test_transform_list = [T.Resize(size=(256, 128)),\
                        T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../images/')
    market1501=Market1501(os.path.join(dataset_dir,'Market-1501-v15.09.15'))
    market_id_num=len(set(list(zip(*market1501.trainSet))[1]))
    sampler=partial(RandomIdSampler,num_instances=4)
    market_train_iter,market_query_iter,market_gallery_iter=load_dataset(market1501,16,32, \
        sampler=sampler,train_transforms=train_transform_list,test_transforms=test_transform_list)
    dukemtmc=DukeMTMC(os.path.join(dataset_dir,'DukeMTMC'))
    _=list(zip(*dukemtmc.trainSet))
    duketrainSet=list(zip(_[0],_[2]))
    duke_train_iter=load_train_iter(duketrainSet,pcids_mode='C',batch_size=16,transforms=train_transform_list)
    duke_query_iter,duke_gallery_iter=load_query_or_gallery_iter(dukemtmc.querySet,batch_size=32, \
        transforms=test_transform_list), \
        load_query_or_gallery_iter(dukemtmc.gallerySet,batch_size=32,transforms=test_transform_list)
    net=arn.AdaptReID_model(classifier_input_dim=2048,classifier_output_dim=market_id_num)
    optimizer=pt.optim.Adam(net.parameters(),lr=0.0003,weight_decay=5e-04)
    scheduler=pt.optim.lr_scheduler.MultiStepLR(optimizer,[130,200,250],gamma=0.1)
    num_epochs=250
    loss_rec_func,loss_cls_func,loss_dif_func,loss_mmd_func,loss_ctr_func=ReconstructionLoss('L1'), \
        nn.CrossEntropyLoss(),DifferenceLoss(),MMDLoss(sigma_list=[1, 2, 10]),ContrastiveLoss()
    loss_lsr_func,loss_tri_func=LabelSmoothLoss(market_id_num),TripletHardLoss(margin=0.3)
    train(net,(market_train_iter,duke_train_iter), \
        (loss_rec_func,loss_lsr_func,loss_dif_func,loss_mmd_func,loss_tri_func),optimizer,num_epochs, \
        coeffis=(0.1,0.1,1,0.1,0.1,0.1,0.1),scheduler=scheduler, \
        out_loss_map=[[(6,0),(0,)],[(7,1),(0,)],[(8,),(1,)],[(4,5),(2,)],[(2,3),(2,)],[(5,3),(3,)],[(3,),(4,)]], \
        losses_name=['loss_rec_s','loss_rec_t','loss_cls_s','loss_dif_t','loss_dif_s','loss_mmd','loss_ctr_s'], \
        checkpoint=checkpoint,device=None,use_pcids=['*','SP','*','*','SP'],use_amp=True,)
        # query_iter=market_query_iter,gallery_iter=market_gallery_iter,evaluate=eval_cmc_map,uda_test_who='source')
    test(net,market_query_iter,market_gallery_iter,eval_cmc_map,uda_test_who='source')
    test(net,duke_query_iter,duke_gallery_iter,eval_cmc_map,uda_test_who='target')