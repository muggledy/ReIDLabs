import cv2
import os.path
from code.gog.gog import GOG
from code.gog.set_parameter import get_default_parameter

test=cv2.imread(os.path.join(os.path.dirname(__file__),'test.bmp'))
param=get_default_parameter(3)
x1=GOG(test,param)
print(x1.shape)

print(x1)
