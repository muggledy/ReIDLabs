import numpy as np
import os.path
# from colorama import init,Fore,Back,Style
import glob
import re

class Market1501:
    def __init__(self,dataset_dir,trainRelabel=True,queryRelabel=False,galleryRelabel=False):
        dataset_dir=os.path.normpath(dataset_dir)
        self.dataset_dir=dataset_dir
        subdirs=['./bounding_box_train','./query','./bounding_box_test']
        self.train_dir=os.path.join(dataset_dir,subdirs[0])
        self.query_dir=os.path.join(dataset_dir,subdirs[1])
        self.gallery_dir=os.path.join(dataset_dir,subdirs[2])

        exist_subdirs=np.array([os.path.exists(i) for i in [self.train_dir,self.query_dir,self.gallery_dir]])
        if not np.all(exist_subdirs):
            raise IOError('Market1501(%s) is not available!'% \
                (', '.join((np.array(subdirs)[np.where(exist_subdirs==False)]).tolist())))

        self.trainRelabel,self.queryRelabel,self.galleryRelabel=trainRelabel,queryRelabel,galleryRelabel
        self.trainSet,self.trainPids,self.trainCids= \
            self.process_dir(self.train_dir,self.trainRelabel) #do relabel is helpful for one-hot coding
        self.querySet,self.queryPids,self.queryCids=self.process_dir(self.query_dir,self.queryRelabel)
        self.gallerySet,self.galleryPids,self.galleryCids=self.process_dir(self.gallery_dir,self.galleryRelabel)

    def process_dir(self,dir_path,relabel=True):
        '''Format of image: 0002_c1s1_000451_03.jpg.We only concerns person id and 
           camera id of each image. Func will return 
           [(img0_path,img0_pid,img0_cid),...],unique person ids,unique camera ids'''
        img_paths=glob.glob(os.path.join(dir_path,'*.jpg'))
        pattern_personid_camid=re.compile(r'([-\d]+)_c(\d)')
        pids,cids=set(),set()
        datasets=[]
        if relabel:
            pcids=[]
            for img_path in img_paths:
                pid,cid=map(int,pattern_personid_camid.findall(img_path)[0])
                if pid==-1: continue
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
                if pid==-1: continue
                pids.add(pid)
                cids.add(cid)
                datasets.append((img_path,pid,cid))
        return datasets,pids,cids

    def print_info(self):
        tnumpids=[len(self.trainPids),len(self.queryPids),len(self.galleryPids)]
        tnumsets=[len(self.trainSet),len(self.querySet),len(self.gallerySet)]
        print("Dataset(Market1501) statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(tnumpids[0],tnumsets[0]))
        print("  query    | {:5d} | {:8d}".format(tnumpids[1],tnumsets[1]))
        print("  gallery  | {:5d} | {:8d}".format(tnumpids[2],tnumsets[2]))
        print("  ------------------------------")

if __name__=='__main__':
    t=Market1501(os.path.join(os.path.dirname(__file__),'../../../images/Market-1501-v15.09.15/'))
    t.print_info()
    