'''
提供下载部分ReID数据集、Eigen第三方库等
References:
[1] https://github.com/NEU-Gou/awesome-reid-dataset
[2] https://gitlab.com/libeigen/eigen.git
'''

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../src/'))
from zoo.tools import Crawler,unzip
from lomo.tools import getcwd

crawler=Crawler()

#注意翻墙~
link_eigen='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMnVVUDdkM2puOGd4czFLP2U9NG9PUjRM.mp3'
link_market1501='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMnBfejlaUFVYMHNnOTlMP2U9Z3JZWGN5.mp3'
link_prid2011='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMjBGUWJNWGhSRi14MnJlP2U9YkFCRjNw.mp3'
link_ilids='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMjR6UThvQ2RWVlhQcUx2P2U9SkN6SW12.mp3'
link_3dpes='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMi1kY2s5aFJGT2FtaERmP2U9RzhFWWc2.mp3'
link_cuhk01='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMndySExWNHJrNC1GRkdOP2U9QU5OclEw.mp3'

def download_dateset(dataset,save_dir=None):
    dataset=dataset.lower()
    if save_dir is None:
        # save_dir=getcwd() #默认保存到此函数调用者所在目录下
        save_dir=os.path.dirname(__file__) #默认保存到此脚本所在即/images/目录下
    link=globals().get('link_%s'%dataset)
    if link is not None:
        crawler.get(link,show_bar=True,name=dataset)
        crawler.save(os.path.join(save_dir,dataset)) #注意压缩数据集下载到本地将自动解压，但是原压缩文件不会被代码删除
        unzip(crawler.save_path,save_dir,crawler.save_file_suffix)
    else:
        raise ValueError('Invalid dataset name!')

def download_eigen(save_dir=None): #一定要解压到/src/third-party/目录下
    if save_dir is None:
        save_dir=os.path.join(os.path.dirname(__file__),'../src/third-party/')
    crawler.get(link_eigen,show_bar=True,name='eigen')
    crawler.save(os.path.join(save_dir,'eigen'))
    unzip(crawler.save_path,save_dir,crawler.save_file_suffix)

if __name__=='__main__':
    download_dateset('ilids')
    # download_dateset('cuhk01')
    # download_eigen()