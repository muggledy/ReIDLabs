from code.viper import get_lomo_viper,get_gog_viper
from code.lomo.tools import calc_cmc,plot_cmc
from code.olda import kolda
from code.tools import euc_dist
import numpy as np
import time

t1=time.time()

# --Stage1: get lomo features--
probFea,galFea=get_lomo_viper() #34%
#probFea,galFea=get_gog_viper() #43%
probFea=probFea.T
galFea=galFea.T

labels=list(range(632))
labels[10]=1
labels[21]=1
kolda(probFea,labels)