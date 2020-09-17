import os
import re
import shutil
from tqdm import tqdm #https://www.jb51.net/article/166648.htm
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../src/'))
from deep.data_manager import Ana78,process_dir
from collections import defaultdict
import numpy as np
from functools import reduce
import scipy.io as scio

def split_cam_like_market1501(base_path,subdirs=['bounding_box_train','query',
                          'bounding_box_test'],analyse=Ana78(r'([-\d]+)_c(\d)'),
                          save_dir=None): #将Market1501按照摄像头编号划分。如果save_dir为空，
                                          #默认保存到market1501同目录下的market1501_split_results
                                          #中，analyse是图像名分析器
    base_path=os.path.normpath(base_path)
    if save_dir is None:
        _1,_2=os.path.split(base_path)
        save_dir=os.path.normpath(os.path.join(_1,'%s_split_results'%_2))
    all_imgs=[]
    for subdir in subdirs:
        subdir_path=os.path.normpath(os.path.join(base_path,subdir))
        _=os.listdir(subdir_path)
        all_imgs.extend(list(zip([subdir_path]*len(_),_)))
    print('Processing for %s'%base_path)
    for base,name in tqdm(all_imgs,ncols=80): #https://www.jianshu.com/p/91294e5606f4
        img_path=os.path.join(base,name)
        _=analyse(name)
        if _ is not None:
            cid=_[1]
            save_img_base_path=os.path.join(save_dir,'cam%d'%cid)
            if not os.path.exists(save_img_base_path):
                os.makedirs(save_img_base_path)
            save_img_path=os.path.join(save_img_base_path,name)
            shutil.copy(img_path,save_img_path)
    print('SPLIT OVER!')

def split_cam_6to2(dataset_dir,probe_cam_ids=[6],gallery_cam_ids=[1,2,3,4,5],only_one=False,iupnii=False): #将6摄像头的Market1501简化为双摄像头设置
    #参考：Kernel cross-view collaborative representation based classification for person re-identification（2018 Elsevier期刊版）
    print('Treat cam%s as Probe and treat cam%s as Gallery, spliting...'%(str(probe_cam_ids),str(gallery_cam_ids)))
    if only_one in [True,False]:
        only_one=[only_one,only_one]
    dataset_dir=os.path.normpath(dataset_dir)
    train_dir=os.path.join(dataset_dir,'bounding_box_train')
    query_dir=os.path.join(dataset_dir,'query')
    gallery_dir=os.path.join(dataset_dir,'bounding_box_test')
    analyse=Ana78(r'([-\d]+)_c(\d)')
    trainSet=process_dir(train_dir,False,analyse)
    querySet=process_dir(query_dir,False,analyse)
    gallerySet=process_dir(gallery_dir,False,analyse)
    all_used_cam_ids=probe_cam_ids+gallery_cam_ids
    trainSet=[(img,pid,cid) for img,pid,cid in trainSet if cid in all_used_cam_ids]
    querySet=[(img,pid,cid) for img,pid,cid in querySet if cid in all_used_cam_ids]
    gallerySet=[(img,pid,cid) for img,pid,cid in gallerySet if cid in all_used_cam_ids]
    dataset=trainSet+querySet+gallerySet #提取所有使用的指定摄像头下的数据
    tdicts=defaultdict(lambda: defaultdict(list))
    for img,pid,cid in dataset:
        tdicts[pid][cid].append((img,pid,cid))
    pids_num=len(list(tdicts.keys()))
    inProbe_inGallery=[] #记录在probe_cam_ids和gallery_cam_ids中同时有图像的行人id列表，候选列表，
                         #挑选其中一部分行人用于训练集，剩余包括不在此列表中的都作为测试集使用，
                         #考虑到一旦使用不同时在probe和gallery中出现的行人，结果将非常低，所以一般
                         #不使用该设置，（if use person not in inProbe_inGallery as test dataset）
                         #，将标志iupnii置为False，即仅仅将inProbe_inGallery列表中的行人划分一半作为
                         #训练集，另一半作为测试集
    for person in tdicts:
        cams=list(tdicts[person].keys())
        inProbe,inGallery=False,False
        for i in probe_cam_ids:
            if i in cams:
                inProbe=True
        for i in gallery_cam_ids:
            if i in cams:
                inGallery=True
        if inProbe and inGallery:
            inProbe_inGallery.append(person)
    np.random.shuffle(inProbe_inGallery)

    if not iupnii:
        train_pids=inProbe_inGallery[:len(inProbe_inGallery)//2]
        test_pids=inProbe_inGallery[len(inProbe_inGallery)//2:]
    else:
        train_pids=inProbe_inGallery[:pids_num//2]
        test_pids=[]
        for person in tdicts:
            if person not in train_pids:
                test_pids.append(person)

    if not only_one[0]: #如果only_one为True，每个行人只各自挑选一张图像在用于训练和测试的probe、gallery集合中
        train_min_used_imgs=[min([sum([len(tdicts[pid][cid]) for cid in tdicts[pid] if cid in probe_cam_ids]), \
            sum([len(tdicts[pid][cid]) for cid in tdicts[pid] if cid in gallery_cam_ids])]) \
            for pid in train_pids] #用于训练集的行人在双摄像头下各自所含图像数量的最小值
    else:
        train_min_used_imgs=[1]*len(train_pids)
    train_probe_imgs_list,train_gallery_imgs_list=[],[]
    train_probe_pids_list,train_gallery_pids_list=[],[]
    pcount,gcount=0,0
    save_dir=os.path.join(os.path.dirname(dataset_dir),'%s_split_6to2_results'%(os.path.basename(dataset_dir)))
    train_probe_save_dir=os.path.join(save_dir,'train','probe')
    train_gallery_save_dir=os.path.join(save_dir,'train','gallery')
    if not os.path.exists(train_probe_save_dir):
        os.makedirs(train_probe_save_dir)
    if not os.path.exists(train_gallery_save_dir):
        os.makedirs(train_gallery_save_dir)
    #train_probe_pids_file=os.path.join(train_probe_save_dir,'pids.pickle')
    #train_gallery_pids_file=os.path.join(train_gallery_save_dir,'pids.pickle')
    train_probe_pids_file=os.path.join(train_probe_save_dir,'pids.mat')
    train_gallery_pids_file=os.path.join(train_gallery_save_dir,'pids.mat')
    for i,pid in enumerate(train_pids):
        candi_probe=reduce(lambda x,y:x+y,[tdicts[pid][cid] for cid in tdicts[pid] if cid in probe_cam_ids])
        np.random.shuffle(candi_probe)
        candi_probe=candi_probe[:train_min_used_imgs[i]]
        candi_gallery=reduce(lambda x,y:x+y,[tdicts[pid][cid] for cid in tdicts[pid] if cid in gallery_cam_ids])
        np.random.shuffle(candi_gallery)
        candi_gallery=candi_gallery[:train_min_used_imgs[i]]
        train_probe_imgs_list.extend(candi_probe)
        train_gallery_imgs_list.extend(candi_gallery)
        for img,*_ in candi_probe:
            shutil.copy(img,os.path.join(train_probe_save_dir,'%05d.jpg'%pcount))
            pcount+=1
        for img,*_ in candi_gallery:
            shutil.copy(img,os.path.join(train_gallery_save_dir,'%05d.jpg'%gcount))
            gcount+=1
        train_probe_pids_list.extend([pid]*train_min_used_imgs[i])
        train_gallery_pids_list.extend([pid]*train_min_used_imgs[i])
    #with open(train_probe_pids_file,'wb') as f:
    #    pickle.dump(train_probe_pids_list,f)
    #with open(train_gallery_pids_file,'wb') as f:
    #    pickle.dump(train_gallery_pids_list,f)
    scio.savemat(train_probe_pids_file,{'pids':np.array(train_probe_pids_list)})
    scio.savemat(train_gallery_pids_file,{'pids':np.array(train_gallery_pids_list)})
    
    test_probe_imgs_list,test_gallery_imgs_list=[],[]
    test_probe_pids_list,test_gallery_pids_list=[],[]
    pcount,gcount=0,0
    test_probe_save_dir=os.path.join(save_dir,'test','probe')
    test_gallery_save_dir=os.path.join(save_dir,'test','gallery')
    if not os.path.exists(test_probe_save_dir):
        os.makedirs(test_probe_save_dir)
    if not os.path.exists(test_gallery_save_dir):
        os.makedirs(test_gallery_save_dir)
    #test_probe_pids_file=os.path.join(test_probe_save_dir,'pids.pickle')
    #test_gallery_pids_file=os.path.join(test_gallery_save_dir,'pids.pickle')
    test_probe_pids_file=os.path.join(test_probe_save_dir,'pids.mat')
    test_gallery_pids_file=os.path.join(test_gallery_save_dir,'pids.mat')
    for pid in test_pids:
        probe_imgs=reduce(lambda x,y:x+y,[tdicts[pid][cid] for cid in tdicts[pid] if cid in probe_cam_ids],[])
        gallery_imgs=reduce(lambda x,y:x+y,[tdicts[pid][cid] for cid in tdicts[pid] if cid in gallery_cam_ids],[])
        if only_one[1]:
            np.random.shuffle(probe_imgs)
            probe_imgs=probe_imgs[:1]
            np.random.shuffle(gallery_imgs)
            gallery_imgs=gallery_imgs[:1]
        test_probe_imgs_list.extend(probe_imgs)
        test_gallery_imgs_list.extend(gallery_imgs)
        for img,*_ in probe_imgs:
            shutil.copy(img,os.path.join(test_probe_save_dir,'%05d.jpg'%pcount))
            pcount+=1
        for img,*_ in gallery_imgs:
            shutil.copy(img,os.path.join(test_gallery_save_dir,'%05d.jpg'%gcount))
            gcount+=1
        test_probe_pids_list.extend([pid]*len(probe_imgs))
        test_gallery_pids_list.extend([pid]*len(gallery_imgs))
    #with open(test_probe_pids_file,'wb') as f:
    #    pickle.dump(test_probe_pids_list,f)
    #with open(test_gallery_pids_file,'wb') as f:
    #    pickle.dump(test_gallery_pids_list,f)
    scio.savemat(test_probe_pids_file,{'pids':np.array(test_probe_pids_list)})
    scio.savemat(test_gallery_pids_file,{'pids':np.array(test_gallery_pids_list)})
    print('SPLIT OVER')

if __name__=='__main__':
    path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'./Market-1501-v15.09.15/')
    split_cam_like_market1501(path,['query','bounding_box_test'])
    # split_cam_6to2(path,only_one=[False,True])