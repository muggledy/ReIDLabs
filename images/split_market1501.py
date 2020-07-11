import os
import re
import shutil
from tqdm import tqdm #https://www.jb51.net/article/166648.htm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../src/'))
from deep.data_manager import Ana78

def split_cam_like_market1501(base_path,subdirs=['bounding_box_train','query',
                          'bounding_box_test'],analyse=Ana78(r'([-\d]+)_c(\d)'),
                          save_dir=None): #如果save_dir为空，默认保存到market1501同目录下
                                          #的market1501_split_results种，analyse是图像名分析器
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
    
if __name__=='__main__':
    path=os.path.join(os.path.dirname(__file__),'./Market-1501-v15.09.15/')
    split_cam_like_market1501(path,['query','bounding_box_test'])