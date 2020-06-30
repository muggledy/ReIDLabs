import torch.nn as nn
import torch as pt
from models.utils import euc_dist,hard_sample_mining,dist_DMLI

class TripletHardLoss(nn.Module):
    '''难样本的三元组损失'''
    def __init__(self,margin):
        super(TripletHardLoss,self).__init__()
        self.margin=margin
        self.ranking_loss=nn.MarginRankingLoss(margin=margin) #MarginRankingLoss(x1,x2,y)=max(-y*(x1-x2)+margin,0), y=1 or -1
                                                              #如果x1,x2,y不是标量，最后会求平均损失
                                                              #Note(Triplet Loss):loss(ap,an)=max(ap-an+margin,0), where ap is 
                                                              #the dist between anchor and positive and an dist between anchor 
                                                              #and negative

    def forward(self,X,Y): #X's shape:(batch_size,dim)，这个批量的大小为PxK，P个不同行人，每个行人K张不同图像
                           #，具体的示例，设K=4，则第1到第4张图像是属于某一行人的四张不同图像，第5到第8张是属于
                           #另一行人的四张不同图像，等等。我们要做的就是从这个批量中挖掘困难样本的三元组。参数Y
                           #的作用仅仅是构建mask掩码（Y是批量数据X的标签值，此处也就是图像的pid），由于我们的批
                           #量是“有序”的（有序是指从前到后每k个图像属于同一行人），实际上Y也可以不需要，有则鲁棒
        dist=euc_dist(X.t())
        dist=dist.clamp(min=1e-12).sqrt() #tensor.clamp(min)用于设置张量数据的下限，即小于min的全部置为min，之后做sqrt是因为
                                          #euc_dist返回的是欧氏距离的平方，其实也可以不做sqrt
        n=X.size(0)
        mask=Y.expand(n,n).eq(Y.expand(n,n).t()) #mask作用以及困难样本挖掘请参见https://www.bilibili.com/video/BV1Pg4y1q7sN?p=36
        dist_ap,dist_an=[],[]
        for i in range(n):
            dist_ap.append(dist[i][mask[i]==True].max().unsqueeze(0)) #在相同行人中找最大距离作为正例困难样本对距离
            dist_an.append(dist[i][mask[i]==False].min().unsqueeze(0)) ##在不同行人中找最小距离作为负例困难样本对距离
        dist_ap=pt.cat(dist_ap)
        dist_an=pt.cat(dist_an)
        return self.ranking_loss(dist_ap,dist_an,-pt.ones_like(dist_ap))
        #反向传播：在计算得到距离矩阵后，事实上，矩阵中的每一个元素都是一个节点，前后节点有连边，并且沿着连边进行反向传播，
        #通过挖掘困难样本，即在距离矩阵中挑选一些节点最后计算损失，显然损失本身也是一个节点而且是最后一个节点，它和之前所有
        #挑选出的距离节点相连，最终反向传播将只会更新这些和损失节点存在连边的距离节点

class AlignedTriLoss(nn.Module):
    '''AlignedReID有两个难样本三元组损失，特别地，全局分支上的难样本挖掘结果共享给局
       部分支，论文实验表明如果各自为政，两分支分别挖掘难样本，会导致网络梯度更新困难'''
    def __init__(self,margin=0.3):
        super(AlignedTriLoss,self).__init__()
        self.margin=margin
        self.ranking_loss_g=nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_l=nn.MarginRankingLoss(margin=margin)

    def forward(self,glfea,targets): #输入是(全局特征,局部特征)和标签值，具体形状参见定义的ResNet50_Aligned模型
        gf,lf=glfea
        dist=euc_dist(gf.t()).sqrt()
        dist_ap_g,dist_an_g,(_,ap_y),(_,an_y)=hard_sample_mining(dist,targets,True)
        global_loss=self.ranking_loss_g(dist_ap_g,dist_an_g,-pt.ones_like(dist_ap_g))
        lf_hard_pos=lf[ap_y] #根据全局分支提供的困难样本索引，获取局部特征困难正例样本数据
        lf_hard_neg=lf[an_y] #获取局部特征困难负例样本
        dist_ap_l=dist_DMLI(lf,lf_hard_pos) #计算正例样本对stripe对齐后的局部特征距离
        dist_an_l=dist_DMLI(lf,lf_hard_neg) #负例样本对对齐后间距
        local_loss=self.ranking_loss_l(dist_ap_l,dist_an_l,-pt.ones_like(dist_ap_l))
        return [global_loss,local_loss] #注意该损失函数返回了两个0维tensor，两个子损失值。加不加括号都一样

if __name__=='__main__':
    target=pt.Tensor([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8])
    features=pt.Tensor(32,2048)
    local_features=pt.randn(32,128,8)
    b=AlignedTriLoss()
    gl,ll=b((features,local_features),target)
    print(gl,ll)