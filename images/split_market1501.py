import os
import re
import shutil
from tqdm import tqdm #https://www.jb51.net/article/166648.htm
import random
class Ana78: #2020.7.8
    '''输入一个图像名称，根据正则模式匹配并返回其摄像头编号（整型值），
       如果名称不含摄像头编号，表示非所需图像文件，或者因为其它一些
       原因要丢弃该文件，则令返回None'''
    def __init__(self,pattern=re.compile(r'([-\d]+)_c(\d)')):
        self.pattern=pattern
        
    def get_cam_id(self,img_name): #由于该类起初是为split_cam_like_market1501设计
                                   #，若不符合具体要求，则继承重载该函数
        ret=self.pattern.findall(img_name)
        if not ret: #模式匹配名称结果为空
            return None
        pid,cid=list(map(int,ret[0]))
        if pid>=0:
            return cid
        else:
            return None
        
    def __call__(self,img_name):
        return self.get_cam_id(img_name)

def split_cam_like_market1501(base_path,subdirs=['bounding_box_train','query',
                          'bounding_box_test'],analyse=Ana78(),
                          save_dir=None): #如果save_dir为空，默认保存到market1501同目录下
                                          #的market1501_split_results种，analyse是图像名分析器
    if save_dir is None:
        _1,_2=os.path.split(os.path.normpath(base_path))
        save_dir=os.path.normpath(os.path.join(_1,'%s_split_results'%_2))
    for subdir in subdirs:
        subdir_path=os.path.normpath(os.path.join(base_path,subdir))
        print('Now processing (sub)dir %s'%subdir_path)
        all_imgs=os.listdir(subdir_path)
        #pbar = tqdm(total=len(all_imgs),ascii=True) ##
        #for img in all_imgs: ##
        for img in tqdm(all_imgs,ncols=80): #https://www.jianshu.com/p/91294e5606f4
            img_path=os.path.join(subdir_path,img)
            cid=analyse(img)
            if cid is not None:
                save_subcamdir_path=os.path.normpath(os.path.join(save_dir,'cam%d'%cid))
                if not os.path.exists(save_subcamdir_path):
                    os.makedirs(save_subcamdir_path)
                save_img_path=os.path.join(save_subcamdir_path,img)
                shutil.copy(img_path,save_img_path)
                #pbar.set_postfix(img=img) ##
            #pbar.update(1) ##
        #pbar.close() ##
    print('OVER!')
    
if __name__=='__main__':
    path=os.path.join(os.path.dirname(__file__),'./Market-1501-v15.09.15/')
    split_cam_like_market1501(path,)