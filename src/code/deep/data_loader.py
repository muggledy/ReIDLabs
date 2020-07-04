from PIL import Image
import os.path
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T

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

def load_dataset(dataset,train_batch_size=None,test_batch_size=None,train_transforms=None, \
                 test_transforms=None,sampler=None,num_workers=None,notrain=False):
    '''dataset是类似Market1501那样的对象，或者说必须具有相同格式的trainSet,querySet和gallerySet属性。传递
       sampler参数（譬如RandomIdSampler）的时候要注意，值必须是partial(RandomIdSampler,num_instances=k)，
       即在传递RandomIdSampler类的时候绑定一些其他必需参数，而这些参数又不好直接传递给load_dataset，因为这样
       的话必须在load_dataset函数定义中添加一项新参数num_instances，不方便且不合理'''
    if not notrain: #如果notrain为True，则不会返回训练数据集批量生产器
        if train_transforms is None:
            train_transforms=[T.Resize((256,128)),T.RandomHorizontalFlip(),T.ToTensor(), \
                T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))]
        train_transforms=T.Compose(train_transforms)
    if test_transforms is None:
        test_transforms=[T.Resize((256,128)),T.ToTensor(), \
            T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))]
    test_transforms=T.Compose(test_transforms)
    
    if num_workers is None:
        num_workers=4
    if not notrain:
        train_batch_size=32 if train_batch_size is None else train_batch_size
        train_iter=DataLoader(reidDataset(dataset.trainSet,train_transforms),batch_size=train_batch_size, \
            shuffle=True if sampler is None else False,sampler= \
            None if sampler is None else sampler(dataset.trainSet),num_workers=num_workers,drop_last=True)
    test_batch_size=(32,32) if test_batch_size is None else ((test_batch_size,test_batch_size) \
        if isinstance(test_batch_size,int) else test_batch_size)
    query_iter=DataLoader(reidDataset(dataset.querySet,test_transforms),batch_size=test_batch_size[0], \
        shuffle=False,num_workers=num_workers,drop_last=False)
    gallery_iter=DataLoader(reidDataset(dataset.gallerySet,test_transforms),batch_size=test_batch_size[1], \
        shuffle=False,num_workers=num_workers,drop_last=False)
    if not notrain:
        return train_iter,query_iter,gallery_iter
    else:
        return query_iter,gallery_iter

if __name__=='__main__':
    import sys
    sys.path.append(os.path.dirname(__file__))
    from data_manager import Market1501
    t=Market1501(os.path.join(os.path.dirname(__file__),'../../../images/Market-1501-v15.09.15/'))
    t.print_info()
    a,b,c=load_dataset(t,32,32)
    print(type(a))