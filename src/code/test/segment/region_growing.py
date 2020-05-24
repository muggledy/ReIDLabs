#comes from: https://github.com/suhas-nithyanand/Image-Segmentation-using-Region-Growing
#Image Segmentation using Region gropwing
#this code has some bug...

import math
from PIL import Image
from pylab import *
import matplotlib.cm as cm
import scipy as sp
import random
import os.path

im = Image.open(os.path.join(os.path.dirname(__file__),'./imgs/3.jpg')).convert('L')
arr = np.asarray(im)

rows,columns = np.shape(arr)
plt.figure()
plt.imshow(im)
plt.gray()
#User selects the intial seed point
print('\nPlease select the initial seed point')

pseed = plt.ginput(1)

x = int(pseed[0][0])
y = int(pseed[0][1])

seed_pixel = []
seed_pixel.append(x)
seed_pixel.append(y)

print('you clicked:',seed_pixel)

plt.close()

img_rg = np.zeros((rows+1,columns+1))
img_rg[seed_pixel[0]][seed_pixel[1]] = 255.0
img_display = np.zeros((rows,columns))

region_points = []
region_points.append([x,y])

def find_region():
	print('\nloop runs till region growing is complete')
	
	count = 0
	x = [-1, 0, 1, -1, 1, -1, 0, 1]
	y = [-1, -1, -1, 0, 0, 1, 1, 1]
	
	while( len(region_points)>0):
		
		if count == 0:
			point = region_points.pop(0)
			i = point[0]
			j = point[1]
		#print('loop runs till length become zero,','len:',len(region_points))
		
		val = arr[i][j]
		lt = val - 8
		ht = val + 8
		
		for k in range(8):
			
			if img_rg[i+x[k]][j+y[k]] !=1:
				try:
					if  arr[i+x[k]][j+y[k]] > lt and arr[i+x[k]][j+y[k]] < ht:
						img_rg[i+x[k]][j+y[k]]=1
						p = [0,0]
						p[0] = i+x[k]
						p[1] = j+y[k]
						if p not in region_points: 
							if 0< p[0] < rows and 0< p[1] < columns:
								''' adding points to the region '''
								region_points.append([i+x[k],j+y[k]])
					else:
						img_rg[i+x[k]][j+y[k]]=0
				except IndexError:
							continue

		point = region_points.pop(0)
		i = point[0]
		j = point[1]
		count = count +1
		
find_region()

plt.figure()
plt.imshow(img_rg, cmap="Greys_r")
#plt.colorbar()
plt.show()
