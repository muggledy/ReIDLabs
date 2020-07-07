import torch as pt
from models import Generator
from datasets import ImageDataset
from torch.utils.data import DataLoader
import numpy as np
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from gog.utils import plot_patches
from main import linear_stretch_img

def test(dataloader,state_savedir_path):
    netG_A2B_path=os.path.join(state_savedir_path,'netG_A2B.pth')
    netG_B2A_path=os.path.join(state_savedir_path,'netG_B2A.pth')
    device=pt.device('cuda' if pt.cuda.is_available() else 'cpu')
    netG_A2B=Generator(3,3).to(device)
    netG_B2A=Generator(3,3).to(device)
    netG_A2B.load_state_dict(pt.load(netG_A2B_path))
    netG_B2A.load_state_dict(pt.load(netG_B2A_path))
    realA_imgs,realB_imgs,fakeB_imgs,fakeA_imgs=[],[],[],[]
    with pt.no_grad():
        for batch_imgs in dataloader:
            real_A=batch_imgs['A'].to(device)
            real_B=batch_imgs['B'].to(device)
            realA_imgs.append(linear_stretch_img(real_A.cpu().numpy(),-1,1))
            realB_imgs.append(linear_stretch_img(real_B.cpu().numpy(),-1,1))
            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)
            fakeA_imgs.append(linear_stretch_img(fake_A.cpu().numpy(),-1,1))
            fakeB_imgs.append(linear_stretch_img(fake_B.cpu().numpy(),-1,1))
    realA_imgs=np.concatenate(realA_imgs,axis=0)
    realB_imgs=np.concatenate(realB_imgs,axis=0)
    fakeA_imgs=np.concatenate(fakeA_imgs,axis=0)
    fakeB_imgs=np.concatenate(fakeB_imgs,axis=0)
    all_imgs=np.rollaxis(np.concatenate([realA_imgs,fakeB_imgs,realB_imgs,fakeA_imgs],axis=0),1,4)
    all_imgs=all_imgs.reshape(4,-1,*all_imgs.shape[-3:])
    plot_patches(all_imgs,False,BGRorRGB='RGB')

if __name__=='__main__':
    import torchvision.transforms as T
    from PIL import Image
    data_root=os.path.join(os.path.dirname(__file__),'./datasets/horse2zebra/')
    batch_size=3
    num=20
    transforms = [ T.ToTensor(),
                   T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImageDataset(data_root, transforms=transforms, mode='test', num=num), \
        batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    state_dir=os.path.join(os.path.dirname(__file__),'./out/horse2zebra/')
    test(dataloader,state_dir)