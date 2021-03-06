import torch as pt
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../'))
from zoo.tools import mkdir_if_missing
from zoo.tools import euc_dist_pro as euc_dist_pro_numpy
from zoo.cprint import cprint_err
import numpy as np
from tqdm import tqdm
from torch import nn, autograd

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, X): # X's shape: (batch_size, ...) #由于并不携带参数
                          #，所以并没有必要写成“层”的形式，直接写成函数即可
        return X.view(X.size(0), -1)

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
            os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../data/models_checkpoints/')

    def save(self,states_info_etc,rel_path=None):
        '''目前传入的states_info_etc（字典）必须包含两个字段，一是网络的参数state_dict，二是当前epoch'''
        self.cur_checkpoint_path=rel_path
        print('Save model state to %s'%self.cur_checkpoint_path,end='')
        dirpath,filename=os.path.split(self.cur_checkpoint_path)
        mkdir_if_missing(dirpath)
        pt.save(states_info_etc,self.cur_checkpoint_path)
        print('!')

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
    return (A+D-2*X1.t().mm(X1 if X2 is None else X2)).clamp(min=1e-12) #tensor.clamp(min)用于设置张量数据的下限，即小于min的全部置为min

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
       是一样的，但需要注意的是，此处labels的构成【必须】是PxK的（N=PxK），表示含有
       P个不同的行人ID，每个ID行人有K个图像，否则代码出错，但是并不要求每一行人连续，
       譬如labels=[0,0,1,1,2,2]是合法的，此处P=3，K=2，而labels=[0,1,1,0,2,2]也
       是合法的。return_inds决定是否返回那些困难样本在dist中的位置索引'''
    N=dist.size(0)
    t_labels=labels.expand(N,N)
    mask_pos=t_labels.eq(t_labels.t())
    # mask_neg=t_labels.ne(t_labels.t())
    mask_neg=mask_pos==False
    max_ap_v,max_ap_ind=pt.max(dist[mask_pos].contiguous().view(N,-1),dim=1)
    min_an_v,min_an_ind=pt.min(dist[mask_neg].contiguous().view(N,-1),dim=1)
    if return_inds:
        # _,y=pt.where(mask_pos==True) #我降低了PyTorch版本为1.1.0报错：TypeError: where(): argument 'input' (position 2) must be Tensor, not bool
        #已知的是1.3版本之后torch.where实现才同numpy.where
        y=(mask_pos==True).nonzero()[:,1] #替代方案：https://blog.csdn.net/judgechen1997/article/details/105820709

        x=pt.arange(N)
        y=y.contiguous().view(N,-1)
        ap_x,ap_y=x,y[x,max_ap_ind]
        # _,y=pt.where(mask_neg==True)
        y=(mask_neg==True).nonzero()[:,1]

        y=y.contiguous().view(N,-1)
        an_x,an_y=x,y[x,min_an_ind]
        return max_ap_v,min_an_v,(ap_x,ap_y),(an_x,an_y) #Note that 
                                #     |           |      #dist[ap_x,ap_y] == dist[ap_coords] == max_ap_v 
                                # ap_coords   an_coords  #and 
                                #                        #dist[an_x,an_y] == dist[an_coords] == min_an_v
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

def shortest_dist(dist): #used for AlignedReID
    '''利用动态规划算法计算最短路径，输入dist形状为(m,n)或(m,n,batch_size)，前者将返回一个标量，
       后者返回一个长度为batch_size的向量'''
    if isinstance(dist,pt.Tensor):
        m,n=dist.size()[:2]
    elif isinstance(dist,np.ndarray):
        m,n=dist.shape[:2]
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
                if isinstance(dist,pt.Tensor):
                    t[i][j]=pt.min(t[i-1][j],t[i][j-1])+dist[i][j]
                elif isinstance(dist,np.ndarray):
                    t[i][j]=np.minimum(t[i-1][j],t[i][j-1])+dist[i][j]
    return t[-1][-1]

def dist_DMLI(X,Y): #used for AlignedReID
    '''具体参考AlignedReID论文。计算X(batch_size,dim,m)和Y(batch_size,dim,n)之间的DMLI距离
       ，其中X[i,:,:]代表了第i个行人的m个长度为dim的水平条特征，函数返回D，形状为(batch_size,)
       ，D[i]即X中第i个行人图像和Y中第i个行人图像的局部特征做对齐后所求距离值，其根据最短路径（基于动
       态规划）得到'''
    if isinstance(X,pt.Tensor):
        stripe_dist_mat=euc_dist_pro(X,Y).sqrt() #(batch_size,m,n)
        stripe_dist_mat=(pt.exp(stripe_dist_mat)-1.0)/(pt.exp(stripe_dist_mat)+1.0) #论文中的归一化处理
        D=shortest_dist(stripe_dist_mat.permute(1,2,0)).view(-1) #permute:(batch_size,m,n)->(m,n,batch_size)
    elif isinstance(X,np.ndarray):
        stripe_dist_mat=np.sqrt(euc_dist_pro_numpy(X,Y))
        stripe_dist_mat=(np.exp(stripe_dist_mat)-1.0)/(np.exp(stripe_dist_mat)+1.0)
        D=shortest_dist(np.rollaxis(stripe_dist_mat,0,3)).reshape(-1)
    return D

def calc_dist_DMLI(X,Y,desc=('query','gallery'),if_print=True): #used for AlignedReID
    '''return a dist matrix(m,n) between X(m,·,·) and Y(n,·,·), about (,·,·), see dist_DMLI'''
    if if_print:
        print('Calc DMLI distance between %s%s and %s%s:'%(desc[0],str(tuple(X.shape)) if \
            isinstance(X,pt.Tensor) else str(X.shape), \
            desc[1],str(tuple(Y.shape)) if isinstance(Y,pt.Tensor) else str(Y.shape)))
    if isinstance(X,pt.Tensor):
        dist=pt.zeros(X.shape[0],Y.shape[0])
    elif isinstance(X,np.ndarray):
        dist=np.zeros((X.shape[0],Y.shape[0]))
    for i in tqdm(range(X.shape[0]),ncols=80,ascii=True):
        qf=X[i]
        if isinstance(X,pt.Tensor):
            qf=pt.stack([qf for i in range(Y.shape[0])])
        elif isinstance(X,np.ndarray):
            qf=np.broadcast_to(qf,(Y.shape[0],)+X.shape[1:])
        dist[i]=dist_DMLI(qf,Y)
    return dist

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

def _weights_init(m): #net.apply(*)
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        pt.nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module): #有了这个类，譬如要添加flatten层，就可以这样写：
                              #LambdaLayer(lambda X:X.view(X.size(0),-1))
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ChannelPool(nn.Module): #used for CBAM
    def forward(self, x):
        return pt.cat( (pt.max(x,1)[0].unsqueeze(1), pt.mean(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module): #used for CBAM
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def logsumexp_2d(tensor): #used for CBAM
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = pt.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

def get_device(device):
    if isinstance(device, str):
        if device == 'GPU':
            device=pt.device('cuda')
        elif device == 'CPU':
            device=pt.device('cpu')
    elif isinstance(device, pt.device):
        pass
    elif device == None:
        device=pt.device('cuda' if pt.cuda.is_available() else 'cpu')
    elif isinstance(device,int): #单卡时，要确保数据是放在指定设备上，多卡时（DP），要确保是在“主设备”上
        device=pt.device('cuda',device)
    return device

class ReverseLayerF(autograd.Function):
    '''
    https://github.com/wogong/pytorch-dann/blob/f81f06d97134e6e1372d635931a8d175087d2986/models/functions.py#L4
    '''
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module): #梯度反转层
    def __init__(self,alpha):
        super(GRL,self).__init__()
        self.grl=ReverseLayerF.apply
        self.alpha=alpha

    def forward(self,X):
        return self.grl(X,self.alpha)

if __name__ == "__main__":
    # k,s=seek_ks_3m(24,6)[:2]
    # print(k,s)
    X=np.arange(5*6).reshape(5,-1)
    Y=np.arange(5*6).reshape(5,-1)
    dist=euc_dist_pro(pt.from_numpy(X).to(pt.float32),pt.from_numpy(Y).to(pt.float32))
    print(dist)
    labels=pt.tensor([0,1,1,0,2,2])
    t=hard_sample_mining(dist,labels,True)
    print(t)