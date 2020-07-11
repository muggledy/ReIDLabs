from initial import *
from deep.models.ResNet import ResNet56_jstl

from deep.train import train,setup_seed
from deep.eval_metric import eval_cmc_map
from deep.models.utils import CheckPoint
import torch as pt
import torch.nn as nn

if __name__=='__main__':
    setup_seed(0)
    