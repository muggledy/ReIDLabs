import numpy as np
import os
# from colorama import init,Fore,Back,Style
import re

class Ana78: #2020.7.8
    '''输入一个图像名称，根据正则模式匹配并返回其摄像头编号（整型值）或者其他信息，
       如果名称不含摄像头编号或期望信息，表示非所需图像文件，或者因为其它一些原因
       要丢弃该文件，则令返回None'''
    def __init__(self,pattern=r'([-\d]+)_c(\d)'):
        self.pattern=re.compile(pattern)
        
    def get_info_from_img_name(self,img_name): #由于该类起初是为split_cam_like_market1501设计
                                               #，若不符合具体要求，则继承重载该函数
        ret=self.pattern.findall(img_name)
        if not ret: #模式匹配名称结果为空，非图片文件
            return None
        pid,cid=list(map(int,ret[0]))
        if pid>=0:
            return pid,cid
        else:
            return None
        
    def __call__(self,img_name):
        return self.get_info_from_img_name(img_name)

class DataSetBase:
    def __init__(self,train_dir=None,query_dir=None,gallery_dir=None, \
                 **kwargs): #这些路径参数的格式实际为(dir_path,relabel,analyse)，如果只给出dir_path，
                            #则需要额外给出关键字参数relabel和analyse，它们将作为默认值应用到train_dir,
                            #query_dir,gallery_dir上
        relabel_analyse=(kwargs.get('relabel'),kwargs.get('analyse'))
        if train_dir:
            self.train_dir,self.trainRelabel,analyse=train_dir if \
                isinstance(train_dir,(list,tuple)) and len(train_dir)==3 else (train_dir,*relabel_analyse)
            self.trainSet,self.trainPids,self.trainCids= \
                self.process_dir(self.train_dir,self.trainRelabel,analyse)
        else:
            self.train_dir=None
        if query_dir:
            self.query_dir,self.queryRelabel,analyse=query_dir if \
                isinstance(query_dir,(list,tuple)) and len(query_dir)==3 else (query_dir,*relabel_analyse)
            self.querySet,self.queryPids,self.queryCids=self.process_dir(self.query_dir,self.queryRelabel,analyse)
        else:
            self.query_dir=None
        if gallery_dir:
            self.gallery_dir,self.galleryRelabel,analyse=gallery_dir if \
                isinstance(gallery_dir,(list,tuple)) and len(gallery_dir)==3 else (gallery_dir,*relabel_analyse)
            self.gallerySet,self.galleryPids,self.galleryCids= \
                self.process_dir(self.gallery_dir,self.galleryRelabel,analyse)
        else:
            self.gallery_dir=None

    def process_dir(self,dir_path,relabel,analyse):
        '''此处的analyse仅仅用于获取图片文件名中携带的行人ID、摄像头ID信息，也必须返回
           该信息或者None，譬如Market1501数据集文件名形如0002_c1s1_000451_03.jpg，
           0002是行人ID，c字符打头的是摄像头ID'''
        assert relabel is not None, "relabel can't be None!"
        assert analyse is not None, "analyse can't be None!"
        img_paths=[i for i in sorted(os.listdir(dir_path)) if i.endswith(('.jpg','.png','.bmp'))]
        pids,cids=set(),set()
        datasets=[]
        if relabel: #do relabel is helpful for one-hot coding
            pcids=[]
            for img_path in img_paths:
                _=analyse(img_path)
                if _ is not None: #Market1501的gallery中有部分行人ID为-1的图像，忽略，所以此处设置只有PID>=0时才计入
                    pid,cid=_
                    pcids.append((pid,cid))
                    pids.add(pid)
                    cids.add(cid)
            pids2normPids={pid:ind for ind,pid in enumerate(pids)}
            cids2normCids={cid:ind for ind,cid in enumerate(cids)}
            for img_path,(pid,cid) in zip(img_paths,pcids):
                datasets.append((os.path.normpath(os.path.join(dir_path,img_path)),pids2normPids[pid],cids2normCids[cid]))
            pids=set(pids2normPids.values())
            cids=set(cids2normCids.values())
        else:
            for img_path in img_paths:
                _=analyse(img_path)
                if _ is not None:
                    pid,cid=_
                    pids.add(pid)
                    cids.add(cid)
                    datasets.append((os.path.normpath(os.path.join(dir_path,img_path)),pid,cid))
        return datasets,pids,cids #这边还返回了所有行人ID和摄像头ID，实际上都用不到，可以删除

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
                 analyse=Ana78(r'([-\d]+)_c(\d)')):
        dataset_dir=os.path.normpath(dataset_dir)
        train_dir=os.path.join(dataset_dir,subdirs[0])
        query_dir=os.path.join(dataset_dir,subdirs[1])
        gallery_dir=os.path.join(dataset_dir,subdirs[2])
        
        exist_subdirs=np.array([os.path.exists(i) for i in [train_dir,query_dir,gallery_dir]])
        if not np.all(exist_subdirs):
            raise IOError('Market1501(%s) is not available!'% \
                (', '.join((np.array(subdirs)[np.where(exist_subdirs==False)]).tolist())))

        super(Market1501,self).__init__((train_dir,trainRelabel,analyse), \
            (query_dir,queryRelabel,analyse),(gallery_dir,galleryRelabel,analyse))

class load_query_gallery(DataSetBase):
    def __init__(self,query_dir,gallery_dir,query_num=10,analyse=Ana78(r'([-\d]+)_c(\d)')): #自定义query的最大数量，如果query
                                                           #文件夹下的图像数量大于此数值，则随机挑选，如果设为None，则为全部
        super(load_query_gallery,self).__init__(query_dir=query_dir, \
            gallery_dir=gallery_dir,relabel=False,analyse=analyse)
        if isinstance(query_num,int) and len(self.querySet)>query_num:
            querySet,queryPids,queryCids=[],set(),set()
            for i in np.random.permutation(len(self.querySet))[:query_num]:
                _=self.querySet[i]
                querySet.append(_)
                queryPids.add(_[1]) #注意只有在relabel为False时才正确，否则需要知道映射关系并作修改，但无此必要
                queryCids.add(_[2]) #同上。实际上这个信息一般根本用不到，无用
            self.querySet=querySet
            self.queryPids=queryPids
            self.queryCids=queryCids

class MixDatasSets:
    def __init__(self):
        pass

if __name__=='__main__':
    dataset_dir=os.path.join(os.path.dirname(__file__),'../../images/Market-1501-v15.09.15/')
    data=Market1501(dataset_dir)
    print(data.trainSet[0])