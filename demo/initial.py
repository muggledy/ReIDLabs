import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../src'))
from zoo.tools import euc_dist,print_cmc,cosine_dist,calc_cmc_map
from lomo.tools import getcwd,mah_dist,calc_cmc,plot_cmc
import numpy as np
from zoo.cprint import cprint,cprint_out,cprint_err,cprint_in
import time