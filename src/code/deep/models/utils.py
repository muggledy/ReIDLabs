import torch as pt
import torch.nn as nn
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from tools import mkdir_if_missing,cprint_err
import numpy as np

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, X): # X's shape: (batch_size, ...) #由于并不携带参数
                          #，所以并没有必要写成“层”的形式，直接写成函数即可
        return X.view(X.shape[0], -1)

class Norm1DLayer(nn.Module):
    '''vec/|vec|'''
    def __init__(self):
        super(Norm1DLayer,self).__init__()
    def forward(self,X): # X's shape: (batch_size, dim)
        return 1.*X/(pt.norm(X,2,dim=-1,keepdim=True).expand_as(X)+1e-12)

def print_net_size(net):
    '''计算模型参数量'''
    print('Model size: {:.5f}M'.format(sum(p.numel() for p in net.parameters())/1000000.0))

class CheckPoint:
    def __init__(self,dir_path=None):
        self.dir_path=dir_path if dir_path is not None else \
            os.path.join(os.path.dirname(__file__),'../../../../data/models_checkpoints/')

    def save(self,states_info_etc,rel_path=None):
        '''目前传入的states_info_etc（字典）必须包含两个字段，一是网络的参数state_dict，二是当前epoch'''
        self.cur_checkpoint_path=rel_path
        print('Save model state to %s'%self.cur_checkpoint_path)
        dirpath,filename=os.path.split(self.cur_checkpoint_path)
        mkdir_if_missing(dirpath)
        pt.save(states_info_etc,self.cur_checkpoint_path)

    def load(self,rel_path=None):
        self.cur_checkpoint_path=rel_path
        print('Load model state from %s'%self.cur_checkpoint_path) #don't delete this print
        try:
            self.states_info_etc=pt.load(self.cur_checkpoint_path) #self.states_info_etc只存放最近一次load的数据
                                                                   #，在访问该属性的时候，务必确认self.loaded为真
        except Exception as e:
            # cprint_err(e)
            print('Load failed! Please check if %s exists!'%self.cur_checkpoint_path)
            self.loaded=False
            return None
        self.loaded=True #只要做了一次load，self.loaded就一直会是True
        return self.states_info_etc

    @property
    def cur_checkpoint_path(self):
        if not hasattr(self,'_cur_checkpoint_path'):
            raise AttributeError \
                ('Not assigned checkpoint\'s file path(obj.cur_checkpoint_path=None)!')
        return self._cur_checkpoint_path

    @cur_checkpoint_path.setter
    def cur_checkpoint_path(self,rel_path):
        if rel_path is not None:
            self._cur_checkpoint_path=os.path.normpath(os.path.join(self.dir_path,rel_path))

def euc_dist(X1,X2=None):
    '''欧氏距离矩阵计算（PytTorch版），参见../../tools.py#euc_dist'''
    A=pt.pow(X1.t(),2).sum(dim=1,keepdim=True)
    if X2 is None:
        D=A.t()
    else:
        D=pt.pow(X2,2).sum(dim=0,keepdim=True)
    return (A+D-2*X1.t().mm(X1 if X2 is None else X2)).clamp(min=1e-12)

def euc_dist_pro(X1,X2=None):
    '''euc_dist的增强版，且兼容euc_dist，参见../../tools.py#euc_dist_pro'''
    X1T=pt.transpose(X1,-2,-1)
    A=pt.pow(X1T,2).sum(dim=-1,keepdim=True)
    if X2 is None:
        D=pt.transpose(A,-2,-1)
    else:
        D=pt.pow(X2,2).sum(dim=-2,keepdim=True)
    return (A+D-2*pt.matmul(X1T,X1 if X2 is None else X2)).clamp(min=1e-12)

def hard_sample_mining(dist,labels,return_inds=False):
    '''困难样本挖掘，dist是一个NxN方阵，labels是轴样本ID序列，长度为N，两个轴标记
       是一样的，但需要注意的是，此处labels的构成必须是PxK的（N=PxK），表示含有P个
       不同的行人ID，每个ID行人有K个图像，否则代码出错。return_inds决定是否返回那
       些困难样本位置索引'''
    N=dist.size(0)
    t_labels=labels.expand(N,N)
    mask_pos=t_labels.eq(t_labels.t())
    # mask_neg=t_labels.ne(t_labels.t())
    mask_neg=mask_pos==False
    max_ap_v,max_ap_ind=pt.max(dist[mask_pos].contiguous().view(N,-1),dim=1)
    min_an_v,min_an_ind=pt.min(dist[mask_neg].contiguous().view(N,-1),dim=1)
    if return_inds:
        _,y=pt.where(mask_pos==True)
        x=pt.arange(N)
        y=y.contiguous().view(N,-1)
        ap_x,ap_y=x,y[x,max_ap_ind]
        _,y=pt.where(mask_neg==True)
        y=y.contiguous().view(N,-1)
        an_x,an_y=x,y[x,min_an_ind]
        return max_ap_v,min_an_v,(ap_x,ap_y),(an_x,an_y) #Note that 
                                #     |           |      #dist[ap_x,ap_y] == dist[ap_inds] == max_ap_v 
                                #  ap_inds     an_inds   #and 
                                #                        #dist[an_x,an_y] == dist[an_inds] == min_an_v
    return max_ap_v,min_an_v

class HorizontalPool2d(nn.Module): #水平池化
    def __init__(self,k=1,s=1,pool_type='max'):
        super(HorizontalPool2d,self).__init__()
        self.k,self.s=k,s
        self.type=pool_type

    def forward(self,X): #X's shape is (batch_size,channel,h,w), pooled 
                         #result's shape is (batch_size,channel,h,1)
        if self.type=='max':
            return nn.MaxPool2d(kernel_size=(self.k,X.size(3)),stride=self.s)(X)
        elif self.type=='avg':
            return nn.AvgPool2d(kernel_size=(self.k,X.size(3)),stride=self.s)(X)

def shortest_dist(dist):
    '''利用动态规划算法计算最短路径，输入dist形状为(m,n)或(m,n,batch_size)，前者将返回一个标量，
       后者返回一个长度为batch_size的向量'''
    m,n=dist.size()[:2]
    t=[[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if i==0 and j==0:
                t[i][j]=dist[i][j]
            elif i==0 and j>0:
                t[i][j]=t[i][j-1]+dist[i][j]
            elif i>0 and j==0:
                t[i][j]=t[i-1][j]+dist[i][j]
            else:
                t[i][j]=pt.min(t[i-1][j],t[i][j-1])+dist[i][j]
    return t[-1][-1]

def dist_DMLI(X,Y):
    '''具体参考AlignedReID论文。计算X(batch_size,dim,m)和Y(batch_size,dim,n)之间的DMLI距离
       ，X[i,:,:]代表了第i个行人的m个长度为dim的水平条特征，返回D，形状为(batch_size1,)
       ，D[i]即X中第i个行人图像和Y中第i个行人图像的局部特征做对齐后所求距离值，根据最短路径（基于动
       态规划）得到'''
    stripe_dist_mat=euc_dist_pro(X,Y).sqrt() #(batch_size,m,n)
    stripe_dist_mat=(pt.exp(stripe_dist_mat)-1.0)/(pt.exp(stripe_dist_mat)+1.0) #论文中的归一化处理
    D=shortest_dist(stripe_dist_mat.permute(1,2,0)).view(-1)
    return D

def seek_ks_3m(h,h_,printf=False):
    '''根据3m原则寻找最佳核尺寸和步长，h为输入长度，h_为输出长度，函数除了返回
       最佳尺寸和步长外，还会返回滑动的最大非重叠区域以及滑动过程中的重叠区域
       3m原则：最大化非重叠区域、最小化重叠区域以及最小化步长'''
    sel=[]
    for k in range(1,h+1):
        for s in range(1,h+1):
            if np.floor((h-k)/s+1)==h_:
                sel.append([k,s,k*h_-max(k-s,0)*(h_-1),max(k-s,0)*(h_-1)]) #(尺寸,步长,最大非重叠滑动区域,总重叠区域)
    sel=np.array(sel)
    if printf:
        print('K | S | M1 | M2')
        print(np.squeeze(sel))
        print('-'*15)
    if sel.shape[0]>1:
        sel=sel[sel[:,2]==np.max(sel[:,2])]
        sel=sel[sel[:,3]==np.min(sel[:,3])]
        return np.squeeze(sel[np.argmin(sel[:,1])])
    elif sel.shape[0]==1:
        return np.squeeze(sel)
    else:
        raise ValueError('seek_ks_3m err')

class SqueezeLayer(nn.Module):
    def __init__(self):
        super(SqueezeLayer,self).__init__()
    def forward(self,X):
        return X.squeeze()

def get_rest_params(net,submodules=None): #譬如要获取网络中除第一个卷积层net.conv1和第二个卷积层net.conv2
                                          #之外的参数，此处submodules就设为['conv1','conv2']。这个函数是为
                                          #学习率分层衰减设计的
    used_params=[]
    if submodules is not None: #如果submodules为None，表示仅剔除PReLU
        for sm in submodules:
            used_params+=list(map(id,getattr(net,sm).parameters()))
    prelu_params=[]
    for m in net.modules():
        if isinstance(m,nn.PReLU): #一般不对PReLU参数做衰减
            prelu_params+=list(map(id,m.parameters()))
    rest_params=filter(lambda x:id(x) not in used_params+prelu_params,net.parameters())
    return rest_params

if __name__ == "__main__":
    k,s=seek_ks_3m(24,6)[:2]
    print(k,s)
