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
        self.ids=self.ids_list.keys()
        self.ids_num=len(self.ids)

    def __iter__(self):
        shuffle_ids=np.random.permutation(self.ids_num)
        ret=[]
        for i in shuffle_ids:
            t=self.ids_list[i]
            ret.extend(np.random.choice(t,self.num_instances,replace= \
                False if len(t)>=self.num_instances else True))
        return iter(ret)

    def __len__(self): #就是iter函数返回值的长度
        return self.num_instances*self.ids_num

if __name__=='__main__':
    from data_manager import Market1501
    import os.path
    market1501=Market1501(os.path.join(os.path.dirname(__file__),'../../../images/Market-1501-v15.09.15/'))
    ret=RandomIdSampler(market1501.trainSet,4).__iter__()
    for i in ret[8:12]:
        print(market1501.trainSet[i])