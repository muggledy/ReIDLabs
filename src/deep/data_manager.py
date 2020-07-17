import numpy as np
import os
# from colorama import init,Fore,Back,Style
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from zoo.tools import norm_labels

class Ana78: #2020.7.8
    '''输入一个图像名称，根据正则模式匹配并返回其摄像头编号（整型值）或者其他信息，
       如果名称不含摄像头编号或期望信息，表示非所需图像文件，或者因为其它一些原因
       要丢弃该文件，则令返回None'''
    def __init__(self,pattern=r'([-\d]+)_c(\d)'): #默认模式字符串是为market1501保留的
        self.pattern=re.compile(pattern)
        
    def get_info_from_img_name(self,img_name): #由于该类起初是为split_cam_like_market1501设计
                                               #，若不符合具体要求，则继承并重载该函数
        ret=self.pattern.findall(img_name)
        if not ret: #模式匹配名称结果为空，表示非图片文件
            return None
        pid,cid=list(map(int,ret[0]))
        if pid>=0: #Market1501的gallery中有部分行人ID为-1的图像，忽略，所以此处设置只有PID>=0时才计入
            return pid,cid
        else:
            return None
        
    def __call__(self,img_name):
        return self.get_info_from_img_name(img_name)

def process_dir(dir_path,relabel,analyse):
    '''传入的analyse必须能够获取图片文件名中携带的行人ID、摄像头ID信息，譬如
       Market1501数据集文件名形如0002_c1s1_000451_03.jpg，
       0002是行人ID，c字符打头的是摄像头ID，analyse处理结果应返回(2,1)'''
    assert relabel is not None, "relabel can't be None!"
    assert analyse is not None, "analyse can't be None!"
    all_img_names=[i for i in sorted(os.listdir(dir_path)) if i.endswith(('.jpg','.png','.bmp'))]
    datasets=[]
    if relabel: #do relabel is helpful for one-hot coding（relabel功能同之前编写的norm_labels函数）
        selected_imgs,pcids=[],[]
        upids,ucids=set(),set()
        for img_name in all_img_names:
            _=analyse(img_name)
            if _ is not None:
                pid,cid=_
                selected_imgs.append(img_name)
                pcids.append(_)
                upids.add(pid)
                ucids.add(cid)
        pids2normPids={pid:ind for ind,pid in enumerate(upids)}
        cids2normCids={cid:ind for ind,cid in enumerate(ucids)}
        for img_name,(pid,cid) in zip(selected_imgs,pcids):
            datasets.append((os.path.normpath(os.path.join(dir_path,img_name)),pids2normPids[pid],cids2normCids[cid]))
    else:
        for img_name in all_img_names:
            _=analyse(img_name)
            if _ is not None:
                pid,cid=_
                datasets.append((os.path.normpath(os.path.join(dir_path,img_name)),pid,cid))
    return datasets

class DataSetBase:
    def __init__(self,train_dir=None,query_dir=None,gallery_dir=None, \
                 **kwargs): #这些路径参数的格式实际为(dir_path,relabel,analyse)，如果只给出dir_path，
                            #则需要额外给出关键字参数relabel和analyse，它们将作为默认值全部应用到
                            #train_dir,query_dir,gallery_dir上
        relabel_analyse=(kwargs.get('relabel'),kwargs.get('analyse'))
        if train_dir:
            self.trainDir,self.trainRelabel,self.trainAnalyse=train_dir if \
                isinstance(train_dir,(list,tuple)) and len(train_dir)==3 else (train_dir,*relabel_analyse)
            self.trainSet=process_dir(self.trainDir,self.trainRelabel,self.trainAnalyse)
        else:
            self.trainDir=None
        if query_dir:
            self.queryDir,self.queryRelabel,self.queryAnalyse=query_dir if \
                isinstance(query_dir,(list,tuple)) and len(query_dir)==3 else (query_dir,*relabel_analyse)
            self.querySet=process_dir(self.queryDir,self.queryRelabel,self.queryAnalyse)
        else:
            self.queryDir=None
        if gallery_dir:
            self.galleryDir,self.galleryRelabel,self.galleryAnalyse=gallery_dir if \
                isinstance(gallery_dir,(list,tuple)) and len(gallery_dir)==3 else (gallery_dir,*relabel_analyse)
            self.gallerySet=process_dir(self.galleryDir,self.galleryRelabel,self.galleryAnalyse)
        else:
            self.galleryDir=None

    def print_info(self):
        print("Dataset statistics:")
        print("  --------------------------------------")
        print("  subset    | # pids | # cids | # images")
        print("  --------------------------------------")
        all_imgs_num,all_pids,all_cids=0,set(),set()
        if self.trainDir:
            if self.trainRelabel!=False:
                imgs,pids,cids=list(zip(*process_dir(self.trainDir,False,self.trainAnalyse)))
            else:
                imgs,pids,cids=list(zip(*self.trainSet))
            all_imgs_num+=len(imgs)
            all_pids.update(pids)
            all_cids.update(cids)
            print("  train     |  {:4d}  |   {:2d}   |   {:6d}".format(len(set(pids)),len(set(cids)),len(imgs)))
        if self.queryDir:
            if self.queryRelabel!=False:
                imgs,pids,cids=list(zip(*process_dir(self.queryDir,False,self.queryAnalyse)))
            else:
                imgs,pids,cids=list(zip(*self.querySet))
            all_imgs_num+=len(imgs)
            all_pids.update(pids)
            all_cids.update(cids)
            print("  query     |  {:4d}  |   {:2d}   |   {:6d}".format(len(set(pids)),len(set(cids)),len(imgs)))
        if self.trainDir and self.queryDir:
            tq="  train+prob|  {:4d}  |   {:2d}   |   {:6d}".format(len(all_pids),len(all_cids),all_imgs_num)
        if self.galleryDir:
            if self.galleryRelabel!=False:
                imgs,pids,cids=list(zip(*process_dir(self.galleryDir,False,self.galleryAnalyse)))
            else:
                imgs,pids,cids=list(zip(*self.gallerySet))
            all_imgs_num+=len(imgs)
            all_pids.update(pids)
            all_cids.update(cids)
            print("  gallery   |  {:4d}  |   {:2d}   |   {:6d}".format(len(set(pids)),len(set(cids)),len(imgs)))
        print("  --------------------------------------")
        if self.trainDir and self.queryDir and self.galleryDir:
            print(tq)
        print("  total     |  {:4d}  |   {:2d}   |   {:6d}".format(len(all_pids),len(all_cids),all_imgs_num))
        print("  --------------------------------------")

class Market1501(DataSetBase): #也可用来处理其他类似Market1501且只需要行人ID和摄像头ID的数据集
    def __init__(self,dataset_dir,subdirs=['./bounding_box_train','./query','./bounding_box_test'], \
                 analyse=Ana78(r'([-\d]+)_c(\d)')):
        dataset_dir=os.path.normpath(dataset_dir)
        train_dir=os.path.join(dataset_dir,subdirs[0])
        query_dir=os.path.join(dataset_dir,subdirs[1])
        gallery_dir=os.path.join(dataset_dir,subdirs[2])
        
        exist_subdirs=np.array([os.path.exists(i) for i in [train_dir,query_dir,gallery_dir]])
        if not np.all(exist_subdirs):
            raise IOError('Market1501(%s) is not available!'% \
                (', '.join((np.array(subdirs)[np.where(exist_subdirs==False)]).tolist())))

        super(Market1501,self).__init__((train_dir,True,analyse),(query_dir,False,analyse), \
            (gallery_dir,False,analyse))

class load_query_gallery(DataSetBase):
    def __init__(self,query_dir,gallery_dir,query_num=10,analyse=None): #自定义query的最大数量，如果query文件夹下的图像数
                                                                        #量大于此数值，则随机挑选，如果设为None，则为全部
        super(load_query_gallery,self).__init__(query_dir=query_dir, \
            gallery_dir=gallery_dir,relabel=False,analyse=analyse)
        if isinstance(query_num,int) and len(self.querySet)>query_num:
            querySet=[]
            for i in np.random.permutation(len(self.querySet))[:query_num]:
                querySet.append(self.querySet[i])
            self.querySet=querySet
            # self.querySet=np.random.choice(self.querySet,query_num,replace=False) #error example

class Ana716(Ana78): #阵对图像名中只含行人ID的解析器
    def get_info_from_img_name(self,img_name):
        ret=self.pattern.findall(img_name)
        if not ret:
            return None
        pid=list(map(int,ret))[0]
        return pid,None #虽然只有行人ID，但仍返回摄像头ID（None），加上仅仅是为了兼容已有代码，少写部分代码

def process_viper(dataset_dir): #注意所有的process_<dataset>都是做了relabel的
    analyse=Ana716(r'(\d{,3})_')
    dataset_a=process_dir(os.path.join(dataset_dir,'./cam_a'),False,analyse)
    dataset_b=process_dir(os.path.join(dataset_dir,'./cam_b'),False,analyse)
    imgs_a,pids_a,_=list(zip(*dataset_a))
    imgs_b,pids_b,_=list(zip(*dataset_b))
    imgs=imgs_a+imgs_b
    pids=pids_a+pids_b
    cids=[0]*len(imgs_a)+[1]*len(imgs_b)
    dataset=list(zip(imgs,norm_labels(pids),norm_labels(cids)))
    return dataset

def process_cuhk01(dataset_dir): #为避免错误，可通过/images/download_dataset.py下载这些数据集
    analyse=Ana716(r'^(\d{,4})')
    dataset=process_dir(dataset_dir,True,analyse)
    imgs,pids,_=list(zip(*dataset))
    dataset=list(zip(imgs,pids,[0,0,1,1]*(int(len(dataset)/4))))
    return dataset

def process_cuhk03(dataset_dir):
    analyse=Ana78(r'(\d+)_c(\d)')
    dataset,upids,ucids=[],set(),set()
    dataset_pair1=process_dir(os.path.join(dataset_dir,'pair1'),True,analyse)
    dataset.extend(dataset_pair1)
    _,pids_pair1,cids_pair1=list(zip(*dataset_pair1))
    upids.update(pids_pair1)
    ucids.update(cids_pair1)
    for i in range(2,6):
        dataset_pairi=process_dir(os.path.join(dataset_dir,'pair%d'%i),True,analyse)
        imgs_pairi,pids_pairi,cids_pairi=list(zip(*dataset_pairi))
        pids_pairi=(np.array(pids_pairi)+max(upids)+1).tolist()
        cids_pairi=(np.array(cids_pairi)+max(ucids)+1).tolist()
        dataset_pairi=list(zip(imgs_pairi,pids_pairi,cids_pairi))
        dataset.extend(dataset_pairi)
        upids.update(pids_pairi)
        ucids.update(cids_pairi)
    return dataset

def process_market1501(dataset_dir):
    analyse=Ana78(r'(\d+)_c(\d)')
    dataset=[]
    for subdir in ['./bounding_box_train','./query','./bounding_box_test']:
        dataset_subdir=process_dir(os.path.join(dataset_dir,subdir),False,analyse)
        dataset.extend(dataset_subdir)
    imgs,pids,cids=list(zip(*dataset))
    pids=norm_labels(pids).tolist()
    cids=norm_labels(cids).tolist()
    dataset=list(zip(imgs,pids,cids))
    return dataset

def process_prid2011(dataset_dir):
    pass

class MixDataSets: #混合多个数据集，当前是为无监督模型迁移编写的，用几个无关数据集进行训练，在目标数据集上测试，
                   #一个好的能够迁移的模型必须能够学习行人的本质特征，即与摄像头差异无关的特征（但这目前做不到）
    def __init__(self,*datasets): #datasets的格式为：('viper','/images/VIPeR.v1.0'),('cuhk01','/images/CUHK01/'),...
        dataset,upids,ucids=[],set(),set()
        self.sub_dataset_num=[]
        for name,path in datasets:
            dataset_i=globals()['process_%s'%name](path)
            imgs_i,pids_i,cids_i=list(zip(*dataset_i))
            if dataset and upids and ucids:
                pids_i=(np.array(pids_i)+max(upids)+1).tolist()
                cids_i=(np.array(cids_i)+max(ucids)+1).tolist()
                dataset_i=list(zip(imgs_i,pids_i,cids_i))
            dataset.extend(dataset_i)
            self.sub_dataset_num.append((name,len(dataset_i)))
            upids.update(pids_i)
            ucids.update(cids_i)
        self.dataset=dataset

    def print_info(self):
        if self.sub_dataset_num:
            names,img_nums=list(zip(*self.sub_dataset_num))
            ends=np.cumsum(np.array(img_nums))
            starts=ends-np.array(img_nums)
        else:
            names=[]
        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset         | # pids | # cids | # images")
        print("  -------------------------------------------")
        total_imgs_num,total_pids_num,total_cids_num=0,0,0
        for i,name in enumerate(names):
            imgs,pids,cids=list(zip(*self.dataset[starts[i]:ends[i]]))
            pids_num=len(set(pids))
            cids_num=len(set(cids))
            imgs_num=len(imgs)
            total_pids_num+=pids_num
            total_cids_num+=cids_num
            total_imgs_num+=imgs_num
            print("  {}     |  {:4d}  |   {:2d}   |   {:6d}".format(name.ljust(10,' '),pids_num,cids_num,imgs_num))
        print("  -------------------------------------------")
        print("  total          |  {:4d}  |   {:2d}   |   {:6d}".format(total_pids_num,total_cids_num,total_imgs_num))
        print("  -------------------------------------------")

if __name__=='__main__':
    dataset_dir=os.path.join(os.path.dirname(__file__),'../../images/')
    mixdatasets=MixDataSets(('cuhk03',os.path.join(dataset_dir,'cuhk03_images/detected')),('cuhk01',os.path.join(dataset_dir,'CUHK01')), \
        ('market1501',os.path.join(dataset_dir,'Market-1501-v15.09.15')))
    mixdatasets.print_info()