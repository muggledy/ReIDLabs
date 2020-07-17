from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np

class RandomIdSampler(Sampler):
    '''Sampler对象传入DataLoader使用，主要实现iter函数，返回的是数据集的“索引序列”
       ，换句话说，假设不使用Sampler，就相当于iter返回0-(n-1)的有序序列，其中n是数
       据集大小。需要注意的是，一旦使用Sampler，那么DataLoader就不能再做shuffle。
       现在我们的批量生产器每次产生的batch中含有P个行人，每个行人有K张图像，那么iter
       返回的就是类似于a1,a2,a3,a4,b1,b2,b3,b4,...这样的序列（举例而已，假设K=4），
       其中a1,a2,a3,a4都是同一行人，只不过是不同的图像，且传入DataLoader的batchsize
       参数此时必须是K=4的整数倍，假设batchsize=32，那么P=8，每次执行iter都会产生一
       个不同的结果，因为DataLoader不能做shuffle，那么就要在iter中引入shuffle，
       Sampler和DataLoader的联系：
       class DataLoader(object):
           ...
           def __next__(self):
               if self.num_workers == 0: #如果非0，则多线程
                   indices = next(self.sample_iter)  # Sampler
                   batch = self.collate_fn([self.dataset[i] for i in indices]) # Dataset
                   if self.pin_memory:
                       batch = _utils.pin_memory.pin_memory_batch(batch)
                   return batch
       参考：https://www.cnblogs.com/marsggbo/p/11308889.html
       注：本函数最初为TriHard Loss编写'''
    def __init__(self,dataset,num_instances): #num_instances就是K
        self.dataset=dataset #dataset可以是譬如Market1501().trainset
        self.num_instances=num_instances
        self.ids_list=defaultdict(list) #键为行人id，值为序列，代表该id行人在数据集中的所有位置序号
        for ind,(_,pid,_) in enumerate(self.dataset):
            self.ids_list[pid].append(ind)
        self.ids=list(self.ids_list.keys())
        self.ids_num=len(self.ids)

    def __iter__(self):
        shuffle_ids=np.random.permutation(self.ids_num)
        ret=[]
        for i in shuffle_ids:
            t=self.ids_list[self.ids[i]]
            ret.extend(np.random.choice(t,self.num_instances,replace= \
                False if len(t)>=self.num_instances else True))
        return iter(ret)

    def __len__(self): #就是iter函数返回值的长度
        return self.num_instances*self.ids_num

class RandomIdSampler2(Sampler): #相较于RandomIdSampler，扩张epoch数据集，RandomIdSampler
                                 #每个epoch采样图像数据量为num_ids*num_instances，其中num_ids
                                 #是行人id数量，譬如market1501训练集中id数为751，num_instances
                                 #为初始化传入的参数，一般设为4，所以每轮epoch使用的数据量就是
                                 #751*4，而实际上整个训练集图片有12936张。RandomIdSampler2针对
                                 #此问题进行改动，扩张了每一轮epoch使用的数据量，是原来的t倍，
                                 #其中t=num_paddings/num_instances。如果num_paddings设为None
                                 #，那么效用等同于RandomIdSampler，没有区别
    def __init__(self,dataset,num_instances,num_paddings=None,statistics=False): #注意num_paddings必须是num_instances的整数倍
        self.dataset=dataset
        self.num_instances=num_instances
        if num_paddings is None:
            self.num_paddings=num_instances
        else:
            if num_paddings%num_instances!=0:
                raise ValueError('num_paddings must be multiple of num_instances!')
            else:
                self.num_paddings=num_paddings
        self.ids_list=defaultdict(list)
        for ind,(_,pid,_) in enumerate(self.dataset):
            self.ids_list[pid].append(ind)
        self.ids=list(self.ids_list.keys())
        self.ids_num=len(self.ids)
        if statistics: #行人拥有的图像数据统计信息
            T=[]
            for i in self.ids_list.values():
                T.append(len(i))
            T=np.array(T)
            print("Dataset statistics(about the img nums each person has):")
            print("  -----------------------------")
            print("   Max | Min | Avg | Med | Mod ") #最后两个是medium（中值）, mode（众数）
            print("  -----------------------------")
            print("   {:^3d} | {:^3d} | {:^3d} | {:^3d} | {:^3d} ".format \
                (T.max(),T.min(),int(T.mean()),np.sort(T)[int(len(T)/2)], \
                np.argmax(np.bincount(T))))
            print("  -----------------------------")

    def __iter__(self): #实际测试发现，由于很多行人图片实在太少，num_paddings的效用被压制，
                        #譬如market1501，num_instances设为4，num_paddings设为8,12,16,32
                        #等，结果其实还要稍微低点，示例见demo_deep_resnet_trihardloss_market.py
        ret=np.zeros((self.ids_num,self.num_paddings),dtype='int')
        for i in range(self.ids_num):
            t=self.ids_list[self.ids[i]]
            ret[i,:]=np.random.choice(t,self.num_paddings,replace= \
                False if len(t)>=self.num_paddings else True)
        ret=np.split(ret,int(self.num_paddings/self.num_instances),axis=1)
        ret=np.concatenate(ret,axis=0)
        np.random.shuffle(ret)
        return iter(ret.flatten().tolist())

    def __len__(self):
        return self.num_paddings*self.ids_num

if __name__=='__main__':
    from data_manager import Market1501
    import os.path
    market1501=Market1501(os.path.join(os.path.dirname(__file__),'../../images/Market-1501-v15.09.15/'))
    ret=RandomIdSampler2(market1501.trainSet,4,16,statistics=True)
    market1501.print_info()
    print(list(ret))