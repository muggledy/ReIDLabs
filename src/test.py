from code.gog.gog import GOG
from code.gog.set_parameter import get_default_parameter
import cv2
import os.path

img=cv2.imread(os.path.join(os.path.dirname(__file__),'../images/VIPeR.v1.0/cam_a/000_45.bmp'))
param=get_default_parameter(0)
GOG(img,param)