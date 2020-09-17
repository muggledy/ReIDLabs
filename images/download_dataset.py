'''
提供下载部分ReID数据集、Eigen第三方库等
References:
[1] https://github.com/NEU-Gou/awesome-reid-dataset
[2] https://gitlab.com/libeigen/eigen.git
[3] https://www.itdaan.com/so?q=%E8%A1%8C%E4%BA%BA%E9%87%8D%E8%AF%86%E5%88%AB%E6%95%B0%E6%8D%AE%E9%9B%86
[4] https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP
[5] https://github.com/dzzp/argos-back/issues/5
'''

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../src/'))
from zoo.tools import Crawler,unzip
from lomo.tools import getcwd

crawler=Crawler()

#注意翻墙~
link_eigen='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMnVVUDdkM2puOGd4czFLP2U9NG9PUjRM.mp3'
link_market1501='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMnBfejlaUFVYMHNnOTlMP2U9Z3JZWGN5.mp3'
link_prid2011='https://link.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpM1V1YlRKZDlCdWZhRDJxP2U9bnBxd0t1.mp3'
link_ilids='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMjR6UThvQ2RWVlhQcUx2P2U9SkN6SW12.mp3'
link_3dpes='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMi1kY2s5aFJGT2FtaERmP2U9RzhFWWc2.mp3'
link_cuhk01='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpMndySExWNHJrNC1GRkdOP2U9QU5OclEw.mp3'
link_cuhk03='https://link.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpM1RXSTVrQXU1WmU5enlGP2U9eGNBYWJr.mp3' #mat文件，1.6G
link_cuhk03np='https://link.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbmtTYTEtT3NXUFBpM1lBQVJmUnF5UURHMVIwP2U9T2FQYWpk.mp3'
link_shinpuhkan='https://link.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcjdORVNsRGhsRXRhLXF4U2ZGZXJ0LVg3NlE/ZT1SRUJ2aGQ=.mp3'
link_cuhksysu='https://link.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcjdORVNsRGhsRXRieG01YURmbFpPS0RURG8/ZT1RMVJiM3c=.mp3'

def download_dateset(dataset,save_dir=None): #注意压缩数据集下载到本地将自动解压，原压缩文件会被代码删除
    dataset=dataset.lower()
    if save_dir is None:
        # save_dir=getcwd() #默认保存到此函数调用者所在目录下
        save_dir=os.path.dirname(os.path.realpath(__file__)) #默认保存到此脚本所在即/images/目录下。记住，以后凡是出现__file__
                                           #的地方都要套一层os.path.realpath(__file__)，否则__file__得到的都是相对路径，终端
                                           #执行报错。https://www.cnblogs.com/ajaxa/p/9016475.html
                                           #http://www.cppcns.com/jiaoben/python/107466.html
    link=globals().get('link_%s'%dataset)
    if link is not None:
        crawler.get(link,show_bar=True,name=dataset,chunk_size=1024*1024*2,use_proxy=True)
        crawler.save(os.path.join(save_dir,dataset))
        crawler.unzip(save_dir,delete=True)
        crawler.clear_temp()
    else:
        raise ValueError('Invalid dataset name!')

def download_eigen(save_dir=None): #一定要解压到/src/third-party/目录下
    if save_dir is None:
        save_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../src/third-party/')
    crawler.get(link_eigen,show_bar=True,name='eigen',chunk_size=1024*1024*2,use_proxy=True)
    crawler.save(os.path.join(save_dir,'eigen'))
    crawler.unzip(save_dir,delete=True)
    crawler.clear_temp()

if __name__=='__main__':
    # download_dateset('ilids')
    # download_dateset('prid2011')
    download_dateset('cuhksysu')
    # download_eigen()
    # import pickle
    # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'crawler_temp.pickle'),'rb') as f:
    #     print(pickle.load(f))