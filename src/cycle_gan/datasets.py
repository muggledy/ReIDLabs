import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, mode='train', **kwargs):
        self.transform = torchvision.transforms.Compose(transforms)
        self.mode=mode

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

        num=kwargs.get('num')
        if mode=='test' and num:
            if num<=len(self.files_A) and num<=len(self.files_B): #random select num images for test
                self.files_A=np.random.choice(self.files_A,num,replace=False).tolist()
                self.files_B=np.random.choice(self.files_B,num,replace=False).tolist()
            else:
                raise ValueError('num(%d) > images\' length of dir /testA(%d) or /testB(%d)!' \
                    %(num,len(self.files_A),len(self.files_B)))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.mode=='train':
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        elif self.mode=='test':
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))