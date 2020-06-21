import torch as pt
import torch.nn as nn
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from tools import mkdir_if_missing,cprint_err

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

if __name__ == "__main__":
    net=FlattenLayer()
    c=CheckPoint()
    f='ResNet50_Classify.pt.tar'
    # c.save({'state':net.state_dict,'epoch':5},f)
    print(c.load(f))