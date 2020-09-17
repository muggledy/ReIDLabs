import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../src')) #唯一添加到搜索路径的文件目录，防止混乱
from zoo.tools import euc_dist,print_cmc,cosine_dist,calc_cmc_map
from lomo.tools import getcwd,mah_dist,calc_cmc,plot_cmc
import numpy as np
from zoo.cprint import cprint,cprint_out,cprint_err,cprint_in
import time