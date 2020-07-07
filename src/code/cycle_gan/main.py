'''
This CycleGAN comes from https://github.com/aitorzip/PyTorch-CycleGAN
消耗的时长难以接受，对于horse2zebra数据集，batch_size取2，50个epoch就需要6
个小时，此时转换结果仍时有错误
2020.7.6
'''

import torch as pt
from models import Generator,Discriminator
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        pt.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        pt.nn.init.normal_(m.weight.data, 1.0, 0.02)
        pt.nn.init.constant_(m.bias.data, 0.0)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def linear_stretch_img(img,mmin=None,mmax=None):
    if mmin is None:
        mmin=np.min(img)
    if mmax is None:
        mmax=np.max(img)
    return ((img-mmin)/(mmax-mmin)*255).astype('uint8')

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = pt.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(pt.cat(to_return))

def train(dataloader,epoches=200,plot=False,**kwargs):
    '''                GAN1 with G_A2B and D_B: gen from x to y
                            拟合域G_A2B(x)的分布和域y相一致 ↰
    (G_A2B,G_B2A)    ┌――――――――――――――――――――――――――――――――――――┴―――――┐
         MIN MAX L = E_y[log(D_B(y))] + E_x[1-log(D_B(G_A2B(x)))]
          (D_B,D_A)   ↓                        + E_x[log(D_A(x))] + E_y[1-log(D_A(G_B2A(y)))]
                 期望(求平均)                     └――┬―――――――――――――――――――――――――――――――――――――――┘
                                                    ↳ GAN2 with G_B2A and D_A: gen from y to x
                                                          拟合域G_B2A(y)的分布和域x相一致
                     + E_x[|G_B2A(G_A2B(x))-x|_1] + E_y[|G_A2B(G_B2A(y))-y|_1]
                       └――┬――――――――――――――――――――――――――――――――――――――――――――――――――┘
                          ↳ cycle consistency（限制映射）
        update G_A2B and G_B2A: MIN E_x[1-log(D_B(G_A2B(x)))] + E_y[1-log(D_A(G_B2A(y)))]
                                + E_x[|G_B2A(G_A2B(x))-x|_1] + E_y[|G_A2B(G_B2A(y))-y|_1]
                                + E_y[|G_A2B(y)-y|_1] +E_x[|G_B2A(x)-x|_1] 注意额外添加的这两项
                                    表示，譬如G_A2B是将x转换为y，当输入为y时，转换结果应当仍是y
        update D_B: MAX E_y[log(D_B(y))] + E_x[1-log(D_B(G_A2B(x)))]
        update D_A: MAX E_x[log(D_A(x))] + E_y[1-log(D_A(G_B2A(y)))]
    note: x is imgs in dir /trainA, y is imgs in dir /trainB
    '''
    flag='cuda' if pt.cuda.is_available() else 'cpu'
    device=pt.device(flag)
    if flag=='cuda':
        cudnn.benchmark=True
    ### 定义模型
    netG_A2B=Generator(3,3).to(device)
    netD_B=Discriminator(3).to(device)
    netG_B2A=Generator(3,3).to(device)
    netD_A=Discriminator(3).to(device)
    ###初始化模型参数
    save_dir_base=kwargs.get('save_dir_base')
    if save_dir_base is None:
        save_dir_base=os.path.join(os.path.dirname(__file__),'./out')
    save_dir_name=kwargs.get('save_dir_name')
    if save_dir_name is None:
        save_dir_name=time.strftime("%Y-%m-%d %HH%MM%SS",time.localtime())
    save_dir_path=os.path.normpath(os.path.join(save_dir_base,save_dir_name))
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    netG_A2B_path=os.path.join(save_dir_path,'netG_A2B.pth')
    netG_B2A_path=os.path.join(save_dir_path,'netG_B2A.pth')
    netD_A_path=os.path.join(save_dir_path,'netD_A.pth')
    netD_B_path=os.path.join(save_dir_path,'netD_B.pth')
    if os.path.exists(netG_A2B_path) and os.path.exists(netG_B2A_path) and os.path.exists(netD_A_path) \
        and os.path.exists(netD_B_path): #加载之前训练的结果
        print('Load net state from dir %s'%save_dir_path)
        netG_A2B.load_state_dict(pt.load(netG_A2B_path))
        netG_B2A.load_state_dict(pt.load(netG_B2A_path))
        netD_A.load_state_dict(pt.load(netD_A_path))
        netD_B.load_state_dict(pt.load(netD_B_path))
    else:
        netG_A2B.apply(weights_init_normal) #https://blog.csdn.net/dss_dssssd/article/details/83990511
        netD_B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
    ### 定义损失
    criterion_GAN = pt.nn.MSELoss() #此处GAN使用了均方差损失而非交叉熵，该损失能使得训练更加稳定以及生成更高质量的图像
                                    #参考论文：https://arxiv.org/abs/1611.04076
    criterion_cycle = pt.nn.L1Loss()
    criterion_identity = pt.nn.L1Loss()
    ### 定义优化器
    lr=kwargs.get('lr',0.0002)
    optimizer_G = pt.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = pt.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = pt.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))
    ### 定义学习率策略
    lr_scheduler_G = pt.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(epoches, kwargs.get('epoch',0), \
        kwargs.get('decay_epoch',100)).step) #https://blog.csdn.net/zisuina_2/article/details/103258573
    lr_scheduler_D_A = pt.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(epoches, \
        kwargs.get('epoch',0), kwargs.get('decay_epoch',100)).step)
    lr_scheduler_D_B = pt.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(epoches, \
        kwargs.get('epoch',0), kwargs.get('decay_epoch',100)).step)
    ### 训练
    if kwargs.get('batch_size') is None:
        for i in dataloader:
            batch_size=len(i['A'])
            break
    target_real = pt.ones(batch_size,1).to(device)
    target_fake = pt.zeros(batch_size,1).to(device)
    batches=len(dataloader)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    internal = kwargs.get('internal',19)
    last_time=time.time()
    for epoch in range(epoches):
        for batch_num,batch_imgs in enumerate(dataloader):
            real_A=batch_imgs['A'].to(device)
            real_B=batch_imgs['B'].to(device)
            #### Generators A2B and B2A ####
            optimizer_G.zero_grad()
            # Identity loss(G_A2B(B) should equal B if real B is fed and G_B2A(A) should equal A if real A is fed)
            loss_identity_B = criterion_identity(netG_A2B(real_B), real_B)*5.0
            loss_identity_A = criterion_identity(netG_B2A(real_A), real_A)*5.0
            # GAN loss
            fake_B = netG_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(netD_B(fake_B), target_real)
            fake_A = netG_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(netD_A(fake_A), target_real)
            # Cycle loss
            loss_cycle_ABA = criterion_cycle(netG_B2A(fake_B), real_A)*10.0 #注意这边乘以10，是求总和损失时的权重系数
            loss_cycle_BAB = criterion_cycle(netG_A2B(fake_A), real_B)*10.0
            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            optimizer_G.step()
            #### Discriminator A ####
            optimizer_D_A.zero_grad()
            loss_D_real = criterion_GAN(netD_A(real_A), target_real)
            if batch_num%internal==0 and plot:
                fakeAs=np.concatenate(np.rollaxis(fake_A.detach().cpu().numpy(),1,4),axis=1)
            fake_A = fake_A_buffer.push_and_pop(fake_A) #为了减少网络的震荡作者使用了过去训练生成的图像而不是最近生成的一张图像
                                                        #来更新判别器（0.5的概率），为此作者设置了包含之前生成的50张图像的缓冲池
            loss_D_fake = criterion_GAN(netD_A(fake_A.detach()), target_fake)
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()
            optimizer_D_A.step()
            #### Discriminator B ####
            optimizer_D_B.zero_grad()
            loss_D_real = criterion_GAN(netD_B(real_B), target_real)
            if batch_num%internal==0 and plot:
                fakeBs=np.concatenate(np.rollaxis(fake_B.detach().cpu().numpy(),1,4),axis=1)
            fake_B = fake_B_buffer.push_and_pop(fake_B) #同上fake_A_buffer，可以直接删除此句，即不使用缓冲池
            loss_D_fake = criterion_GAN(netD_B(fake_B.detach()), target_fake)
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            if batch_num%internal==0:
                cur_time=time.time()
                print('epoch=%d, batch=[%d/%d], loss(G)=%f, loss(D_B)=%f, loss(D_A)=%f, time(s)=%f'% \
                    (epoch+1,batch_num+1,batches,loss_G,loss_D_B,loss_D_A,cur_time-last_time))
                last_time=cur_time
                if plot: #实时观察训练效果
                    realAs=np.concatenate(np.rollaxis(real_A.cpu().numpy(),1,4),axis=1)
                    realBs=np.concatenate(np.rollaxis(real_B.cpu().numpy(),1,4),axis=1)
                    plt.imshow(linear_stretch_img(np.concatenate([realAs,fakeBs,realBs,fakeAs],axis=0),-1,1))
                    plt.pause(0.01)
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        ### 保存模型参数
        pt.save(netG_A2B.state_dict(),netG_A2B_path)
        pt.save(netG_B2A.state_dict(),netG_B2A_path)
        pt.save(netD_A.state_dict(),netD_A_path)
        pt.save(netD_B.state_dict(),netD_B_path)
        print('Save net state into dir %s'%save_dir_path)
    if plot:
        plt.show()

if __name__=='__main__':
    from torch.utils.data import DataLoader
    from datasets import ImageDataset
    import torchvision.transforms as T
    from PIL import Image
    ### 准备数据：https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
    data_root=os.path.join(os.path.dirname(__file__),'./datasets/summer2winter/')
    batch_size=1
    size=(256,256)
    transforms=[T.Resize([int(1.12*i) for i in size], Image.BICUBIC), 
                T.RandomCrop(size), 
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    data_iter=DataLoader(ImageDataset(data_root,transforms=transforms,mode='train'), \
        batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)
    train(data_iter,plot=True,save_dir_name='summer2winter') #网络参数（4个子网络）保存的位置由两部分构成，一是
                                                           #save_dir_base，二是save_dir_name，都是可省的，默
                                                           #认值分别为'./out/'和'日期时间'