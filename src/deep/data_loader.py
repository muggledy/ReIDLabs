from PIL import Image
import os
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../'))
from deep.transform import RandomErasing,ColorJitter,Lighting
# from zoo.tools import LRU

def read_image(img_path):
    if not os.path.exists(img_path):
        raise IOError('%s doesn\'t exist!'%img_path)
    img=Image.open(img_path).convert('RGB')
    return img

class reidDataset(Dataset):
    def __init__(self,dataset,transform=None):
        self.dataset=dataset
        self.transform=transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        img_path,pid,cid=self.dataset[index]
        img=read_image(img_path)
        if self.transform is not None:
            img=self.transform(img)
        return img,pid,cid

class testDataset(Dataset): #用于构造测试图像数据集，测试图像没有标签也不需要知道任何监督信息
    def __init__(self,dataset,transform=None): #dataset默认应该是一个图像路径列表，但也可以是一个文件夹路径
        if isinstance(dataset,str) and os.path.isdir(dataset):
            all_imgs_file=[i for i in sorted(os.listdir(dataset)) if i.endswith(('.jpg','.png','.bmp'))]
            self.dataset=[os.path.join(dataset,i) for i in all_imgs_file]
        else:
            self.dataset=dataset
        self.transform=transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        img=read_image(self.dataset[index])
        if self.transform is not None:
            img=self.transform(img)
        return (img,)

'''
  DataLoader ←——————————————————— Batch Data
       ↓                              ↑
 DataLoaderIter                   collate_fn
       ↓                              ↑
    Sampler —————→ Index          Img,Label
       ↓             |                ↑
 DataSetFetcher ←————┛                |
       ↓                              |
    Dataset ——————→ getitem ————→ transforms
'''

default_train_transforms=[T.Resize((256,128)),T.RandomHorizontalFlip(),RandomErasing(), \
    T.ToTensor(),T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))] #T.Normalize: 
    #逐channel对图像进行标准化，公式：output = (input - mean) / std。其中mean默认为各通道的均值，std默认为各通道的标准差
default_test_transforms=[T.Resize((256,128)),T.ToTensor(), \
    T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))]

#DataLoader可能是速度瓶颈，参考：https://www.zhihu.com/question/356829360/answer/907832358
#https://github.com/NVIDIA/apex/issues/304
#https://zhuanlan.zhihu.com/p/80695364，prefetch_generator和data_prefetcher实测都没速度提升啊

def load_dataset(dataset,train_batch_size=None,test_batch_size=None,train_transforms=None, \
                 test_transforms=None,sampler=None,num_workers=None,notrain=False):
    '''dataset是类似Market1501那样的对象，或者说必须具有相同格式的trainSet,querySet和gallerySet属性。传递
       sampler参数（譬如RandomIdSampler）的时候要注意，值必须是partial(RandomIdSampler,num_instances=k)，
       即在传递RandomIdSampler类的时候绑定一些其他必需参数，而这些参数又不好直接传递给load_dataset，因为这样
       的话必须在load_dataset函数定义中添加一项新参数num_instances，不方便且不合理'''
    if not notrain: #如果notrain为True，则不会返回训练数据集批量生产器
        if train_transforms is None:
            train_transforms=default_train_transforms
    if test_transforms is None:
        test_transforms=default_test_transforms
    
    #话说pin_memory究竟有什么用？
    #https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-works-in-dataloader
    if num_workers is None:
        num_workers=4
    if not notrain:
        train_batch_size=32 if train_batch_size is None else train_batch_size
        train_iter=DataLoader(reidDataset(dataset.trainSet,T.Compose(train_transforms)),batch_size= \
                train_batch_size,shuffle=True if sampler is None else False,sampler= \
                None if sampler is None else sampler(dataset.trainSet),num_workers=num_workers,drop_last=True)
    test_batch_size=(32,32) if test_batch_size is None else ((test_batch_size,test_batch_size) \
        if isinstance(test_batch_size,int) else test_batch_size)
    query_iter=DataLoader(reidDataset(dataset.querySet,T.Compose(test_transforms)),batch_size=test_batch_size[0], \
        shuffle=False,num_workers=num_workers,drop_last=False)
    gallery_iter=DataLoader(reidDataset(dataset.gallerySet,T.Compose(test_transforms)),batch_size= \
        test_batch_size[1],shuffle=False,num_workers=num_workers,drop_last=False)
    if not notrain:
        return train_iter,query_iter,gallery_iter
    else:
        return query_iter,gallery_iter

def load_train_iter(dataset,batch_size=32,transforms=default_train_transforms,sampler=None,num_workers=4):
    train_iter=DataLoader(reidDataset(dataset,T.Compose(transforms)), \
        batch_size=batch_size,shuffle=True if sampler is None else False,num_workers=num_workers, \
        sampler=None if sampler is None else sampler(dataset),drop_last=True)
    return train_iter

def load_query_or_gallery_iter(dataset,batch_size=32,transforms=default_test_transforms,num_workers=4):
    query_or_gallery_iter=DataLoader(reidDataset(dataset,T.Compose(transforms)), \
        batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False)
    return query_or_gallery_iter

if __name__=='__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from data_manager import Market1501
    t=Market1501(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../images/Market-1501-v15.09.15/'))
    t.print_info()
    a,b,c=load_dataset(t,32,32)
    for imgs,pids,cids in a:
        print(imgs.size(),pids,cids)
        break