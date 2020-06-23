import torch.nn as nn
import torch as pt
from models.utils import euc_dist

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

if __name__=='__main__':
    a=pt.tensor([1,3,2,5])
    b=pt.tensor([2,1,3,4])
    loss=nn.MarginRankingLoss(margin=0)
    print(loss(a.t(),b.t(),-pt.ones_like(a).t()))