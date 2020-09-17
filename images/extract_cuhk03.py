'''
从mat文件提取出CUHK03数据集行人图像
CUHK-03
　　├── "detected" ── 5 x 1 cells （行人框由机器标注）
　　　　　　　├── 843x10 cells （来自摄像头组1，行代表行人，前5列和后5列分别来自组内的两个不同摄像头，个别图像可能空缺）
　　　　　　　├── 440x10 cells （来自摄像头组2，注意不同摄像头组拍摄的行人ID是不重复的）
　　　　　　　├── 77x10 cells （来自摄像头组3）
　　　　　　　├── 58x10 cells （来自摄像头组4）
　　　　　　　├── 49x10 cells （来自摄像头组5）
　　├── "labeled" ── 5 x 1 cells （结构同detected，只不过行人框是由人工标注）
　　　　　　　├── 843x10 cells
　　　　　　　├── 440x10 cells
　　　　　　　├── 77x10 cells
　　　　　　　├── 58x10 cells
　　　　　　　├── 49x10 cells
　　├── "testsets" ── 20 x 1 cells （测试集，由20个100x2的矩阵组成）
　　　　　　　├── 100 x 2 double matrix （100个测试样本，第1列为摄像头组索引，第2列为行人索引，重复测试过程20次取均值）
测试协议[1]
原论文实验中并未使用摄像头组4和5的数据，所以总计有1360个不同的行人，随机选出100个用于测试集（划分见testsets），剩下的选1160个用于
训练集，还剩下100个用于验证集。最后基于single-shot setting（gallery中只一个ground truth）计算得到CMC曲线
测试协议[2]
类似于Market-1501，它将数据集分为包含767个行人的训练集和包含700个行人的测试集。在测试阶段，我们随机选择一张图像作为query，剩下的作
为gallery，对于每个行人，有多个ground truth在gallery中
References:
[1] Li W, Zhao R, Xiao T, et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification[C]// 
    IEEE Conference on Computer Vision and Pattern Recognition. IEEE Computer Society, 2014:152-159.
[2] Zhong Z, Zheng L, Cao D, et al. Re-ranking person re-identification with k-reciprocal encoding[C]//
    Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on. IEEE, 2017: 3652-3661
'''

import os.path
import scipy.io as scio
from PIL import Image
from tqdm import tqdm
import numpy as np

def extract_cuhk03_images(mat_file,save_dir=None):
    if save_dir is None:
        save_dir=os.path.dirname(os.path.realpath(__file__)) #默认保存到此脚本所在即/images/目录下
    save_dir=os.path.join(save_dir,'./cuhk03_images/')
    data=scio.loadmat(mat_file)
    sub_train_tag=['detected','labeled']
    num=0
    for tag in sub_train_tag:
        for pair_num in range(5):
            pair_data=data[tag][pair_num][0]
            for pid in range(pair_data.shape[0]):
                for ind in range(pair_data.shape[1]):
                    if pair_data[pid][ind].size!=0:
                        num+=1
    with tqdm(total=num,ncols=80) as pbar:
        for tag in sub_train_tag:
            for pair_num in range(5):
                pair_data=data[tag][pair_num][0]
                save_sub_train_pair_dir=os.path.join(save_dir,tag,'pair%d'%(pair_num+1))
                if not os.path.exists(save_sub_train_pair_dir):
                    os.makedirs(save_sub_train_pair_dir)
                for pid in range(pair_data.shape[0]):
                    for ind in range(pair_data.shape[1]):
                        if pair_data[pid][ind].size!=0:
                            img_name='%s_c%d_%s.jpg'%(str(pid+1).zfill(4),int(ind/5)+1,str(ind%5+1).zfill(4))
                            Image.fromarray(pair_data[pid][ind]).save(os.path.join(save_sub_train_pair_dir,img_name))
                            pbar.update(1)
    testsets=[]
    for i in range(20):
        testsets.append(data['testsets'][i][0])
    scio.savemat(os.path.join(save_dir,'testsets.mat'),{'testsets':np.stack(testsets)})

if __name__=='__main__':
    extract_cuhk03_images(os.path.join(os.path.dirname(os.path.realpath(__file__)),'./cuhk03/cuhk-03.mat'))