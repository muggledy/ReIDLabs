import numpy as np
import os
# from colorama import init,Fore,Back,Style
import glob
import re

class DataSetBase:
    def __init__(self,train_dir=None,query_dir=None,gallery_dir=None, \
                 **kwargs): #这些路径参数的格式实际为(dir_path,relabel,pattern)
                            #，如果只给出dir_path，则需要额外给出关键字参数
                            #relabel和pattern，它们将作为默认值应用到train_dir,
                            #query_dir,gallery_dir上
        relabel_pattern=(kwargs.get('relabel'),kwargs.get('pattern'))
        if train_dir:
            self.train_dir,self.trainRelabel,pattern=train_dir if \
                isinstance(train_dir,(list,tuple)) and len(train_dir)==3 else (train_dir,*relabel_pattern)
            self.trainSet,self.trainPids,self.trainCids= \
                self.process_dir(self.train_dir,self.trainRelabel,pattern)
        else:
            self.train_dir=None
        if query_dir:
            self.query_dir,self.queryRelabel,pattern=query_dir if \
                isinstance(query_dir,(list,tuple)) and len(query_dir)==3 else (query_dir,*relabel_pattern)
            self.querySet,self.queryPids,self.queryCids=self.process_dir(self.query_dir,self.queryRelabel,pattern)
        else:
            self.query_dir=None
        if gallery_dir:
            self.gallery_dir,self.galleryRelabel,pattern=gallery_dir if \
                isinstance(gallery_dir,(list,tuple)) and len(gallery_dir)==3 else (gallery_dir,*relabel_pattern)
            self.gallerySet,self.galleryPids,self.galleryCids= \
                self.process_dir(self.gallery_dir,self.galleryRelabel,pattern)
        else:
            self.gallery_dir=None

    def process_dir(self,dir_path,relabel,pattern_personid_camid):
        '''模式pattern是干嘛的，对于给定的图片路径，通过正则模式来匹配获取图片文
           件名中携带的行人ID、摄像头ID信息，譬如Market1501数据集文件名形如
           0002_c1s1_000451_03.jpg，0002是行人ID，c字符打头的是摄像头ID，目前我
           们只关注行人ID和摄像头ID，如果你要获取其他信息，则需要重载此函数。函数
           返回[(img0_path,img0_pid,img0_cid),...],unique pids,unique cids'''
        assert relabel is not None, "relabel can't be None!"
        assert pattern_personid_camid is not None, "pattern can't be None!"
        img_paths=glob.glob(os.path.join(dir_path,'*.jpg'))
        pids,cids=set(),set()
        datasets=[]
        if relabel: #do relabel is helpful for one-hot coding
            pcids=[]
            for img_path in img_paths:
                pid,cid=map(int,pattern_personid_camid.findall(img_path)[0])
                if pid>=0: #Market1501的gallery中有部分行人ID为-1的图像，忽略，所以此处设置只有PID>=0时才计入
                    pcids.append((pid,cid))
                    pids.add(pid)
                    cids.add(cid)
            pids2normPids={pid:ind for ind,pid in enumerate(pids)}
            cids2normCids={cid:ind for ind,cid in enumerate(cids)}
            for img_path,(pid,cid) in zip(img_paths,pcids):
                datasets.append((img_path,pids2normPids[pid],cids2normCids[cid]))
            pids=set(pids2normPids.values())
            cids=set(cids2normCids.values())
        else:
            for img_path in img_paths:
                pid,cid=map(int,pattern_personid_camid.findall(img_path)[0])
                if pid>=0:
                    pids.add(pid)
                    cids.add(cid)
                    datasets.append((img_path,pid,cid))
        return datasets,pids,cids #这边还返回了所有行人ID和摄像头ID，实际上用不到，可以删除

    def print_info(self):
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        if self.train_dir:
            print("  train    | {:5d} | {:8d}".format(len(self.trainPids),len(self.trainSet)))
        if self.query_dir:
            print("  query    | {:5d} | {:8d}".format(len(self.queryPids),len(self.querySet)))
        if self.gallery_dir:
            print("  gallery  | {:5d} | {:8d}".format(len(self.galleryPids),len(self.gallerySet)))
        print("  ------------------------------")

class Market1501(DataSetBase): #也可用来处理其他只需要行人ID和摄像头ID的数据集
    def __init__(self,dataset_dir,trainRelabel=True,queryRelabel=False,galleryRelabel=False, \
                 subdirs=['./bounding_box_train','./query','./bounding_box_test'], \
                 pattern_personid_camid=re.compile(r'([-\d]+)_c(\d)')):
        dataset_dir=os.path.normpath(dataset_dir)
        train_dir=os.path.join(dataset_dir,subdirs[0])
        query_dir=os.path.join(dataset_dir,subdirs[1])
        gallery_dir=os.path.join(dataset_dir,subdirs[2])
        
        exist_subdirs=np.array([os.path.exists(i) for i in [train_dir,query_dir,gallery_dir]])
        if not np.all(exist_subdirs):
            raise IOError('Market1501(%s) is not available!'% \
                (', '.join((np.array(subdirs)[np.where(exist_subdirs==False)]).tolist())))

        super(Market1501,self).__init__((train_dir,trainRelabel,pattern_personid_camid), \
            (query_dir,queryRelabel,pattern_personid_camid),(gallery_dir,galleryRelabel,pattern_personid_camid))

class load_query_gallery(DataSetBase):
    def __init__(self,query_dir,gallery_dir,query_num=10): #自定义query的最大数量，如果query文件夹下的图像数
                                                           #量大于此数值，则随机挑选，如果设为None，则为全部
        super(load_query_gallery,self).__init__(query_dir=query_dir, \
            gallery_dir=gallery_dir,relabel=False,pattern=re.compile(r'([-\d]+)_c(\d)'))
        if isinstance(query_num,int) and len(self.querySet)>query_num:
            querySet,queryPids,queryCids=[],set(),set()
            for i in np.random.permutation(len(self.querySet))[:query_num]:
                t=self.querySet[i]
                querySet.append(t)
                queryPids.add(t[1]) #注意只有在relabel为False时才正确，否则需要知道映射关系并作修改，但无此必要
                queryCids.add(t[2]) #同上。实际上这个信息一般根本用不到，无用
            self.querySet=querySet
            self.queryPids=queryPids
            self.queryCids=queryCids

if __name__=='__main__':
    p=os.path.join(os.path.dirname(__file__),'../../../images/Market-1501-v15.09.15/')
    t=Market1501(p)
    t.print_info()
    qg=load_query_gallery(t.query_dir,t.gallery_dir)
    qg.print_info()
    print(qg.querySet)
    print(qg.queryPids)