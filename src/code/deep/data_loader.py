from PIL import Image
import os.path
from torch.utils.data import Dataset,DataLoader
import sys
sys.path.append(os.path.dirname(__file__))
from data_manager import Market1501
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

def load_Market1501(dataset_dir,train_batch_size,test_batch_size, \
                    train_transforms=None,test_transforms=None,num_workers=None):
    if train_transforms is None:
        train_transforms=[T.Resize((256,128)),T.RandomHorizontalFlip(),T.ToTensor(), \
            T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))]
    train_transforms=T.Compose(train_transforms)
    if test_transforms is None:
        test_transforms=[T.Resize((256,128)),T.ToTensor(), \
            T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))]
    test_transforms=T.Compose(test_transforms)
    market1501=Market1501(dataset_dir)
    if num_workers is None:
        num_workers=4
    train_iter=DataLoader(reidDataset(market1501.trainSet,train_transforms),batch_size=train_batch_size, \
        shuffle=True,num_workers=num_workers,drop_last=True)
    query_iter=DataLoader(reidDataset(market1501.querySet,test_transforms),batch_size=test_batch_size, \
        shuffle=False,num_workers=num_workers,drop_last=False)
    gallery_iter=DataLoader(reidDataset(market1501.gallerySet,test_transforms),batch_size=test_batch_size, \
        shuffle=False,num_workers=num_workers,drop_last=False)
    return train_iter,query_iter,gallery_iter,market1501

if __name__=='__main__':
    from data_manager import Market1501
    t=Market1501(os.path.join(os.path.dirname(__file__),'../../../images/Market-1501-v15.09.15/'))
    t.print_info()
    print(t.trainRelabel)
    s=reidDataset(t.trainSet)
    print(s[100]) #s[100][0].show()