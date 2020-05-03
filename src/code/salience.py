'''
forked from https://github.com/appstud/person-reidentification-patch-based
'''

import scipy.io as sio
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from skimage import feature
import numpy as np
import os
import imutils
from os import listdir
import pdb
import time
import multiprocessing as mp

def calculateColorHistogramsOfPatch(patch,nbins=32):
    ###calculate the 3 channel histogram of a patch and normalize it
    
    colorDescriptor=[]
    for i in range(patch.shape[2]):
        hist,_=np.histogram(patch[:,:,i].ravel(),bins=32,range=(0,255))
        #hist=hist/(np.linalg.norm(hist)+1e-7)
        #hist=hist/(np.linalg.norm(hist)+1e-7)
        colorDescriptor.append(hist)
        #print(np.sum(hist**2))
    colorDescriptor=np.asarray(colorDescriptor)/(np.linalg.norm(np.asarray(colorDescriptor))+1e-7)
    return np.array(colorDescriptor).reshape(-1)

def calculateColorHistogramOfAllPatchsInImage(image, grid, patchSize, nbins=32):
    ### Grid: a list of 2 list: The first one corresponds to the x points that corresponds to the center of each patch in the image while the second correspond to its y coordinate
    ### patchSize: an int representing the size of each square patch (width or height)
    ### nbins: number of bins for histogram

    height,width,numChannels=image.shape    
    x_grid=grid[0]
    y_grid=grid[1]   
    Nx=len(x_grid)
    Ny=len(y_grid)
    colorDescriptor=np.zeros([numChannels*nbins,Nx*Ny])
    
    for j,y in enumerate(y_grid):
        for i,x in enumerate(x_grid):         
            patch=image[int(y-patchSize/2.0):int(y+patchSize/2.0),int(x-patchSize/2.0):int(x+patchSize/2.0)]
            colorDescriptor[:,j*Nx+i]=calculateColorHistogramsOfPatch(patch,nbins=nbins)
            
    return colorDescriptor

def parallelColorHistogramOfAllPatchsInImageAtDifferentScales(image, grid_step, patchSize, nbins=32,scales=[1,0.75,0.5]):
    ### Grid: a list of 2 list: The first one corresponds to the x points that corresponds to the center of each patch in the image while the second correspond to its y coordinate
    ### patchSize: an int representing the size of each square patch (width or height)
    ### nbins: number of bins for histogram
    ### gridStep: the grid step
    
    pool = mp.Pool(3)
    height,width,numChannels=image.shape
    
    Nx=len(np.arange(patchSize/2,width-patchSize/2+1,grid_step))
    Ny=len(np.arange(patchSize/2,height-patchSize/2+1,grid_step))
    
    colorDescriptor=None
    resizedImageList=[]
    x_gridList=[]
    y_gridList=[]
    patchSizeAtCurrentScaleList=[]
    for scale in scales:
        
        patchSizeAtCurrentScale=patchSize*scale
        x_grid=np.linspace(patchSizeAtCurrentScale/2,width*scale-patchSizeAtCurrentScale/2+1,Nx)
        y_grid=np.linspace(patchSizeAtCurrentScale/2,height*scale-patchSizeAtCurrentScale/2+1,Ny)
        if(scale!=1):
            resized = imutils.resize(image, width = int(image.shape[1]*scale))
            
        else:
            resized=image
        resizedImageList.append(resized)
        x_gridList.append(x_grid)
        y_gridList.append(y_grid)
        patchSizeAtCurrentScaleList.append(patchSizeAtCurrentScale)
            
    results = [pool.apply(calculateColorHistogramOfAllPatchsInImage, args=(resizedImageList[i], [x_gridList[i],y_gridList[i]], patchSizeAtCurrentScaleList[i])) for i  in range(len(patchSizeAtCurrentScaleList))]     
       
    for desc in results:
        if colorDescriptor is None:
            colorDescriptor=desc
        else:
            colorDescriptor=np.vstack((colorDescriptor,desc))
    
    return colorDescriptor
    
    
def calculateColorHistogramOfAllPatchsInImageAtDifferentScales(image, grid_step, patchSize, nbins=32,scales=[1,0.75,0.5]):
    ### Grid: a list of 2 list: The first one corresponds to the x points that corresponds to the center of each patch in the image while the second correspond to its y coordinate
    ### patchSize: an int representing the size of each square patch (width or height)
    ### nbins: number of bins for histogram
    ### gridStep: the grid step
    
    height,width,numChannels=image.shape
    
    Nx=len(np.arange(patchSize/2,width-patchSize/2+1,grid_step))
    Ny=len(np.arange(patchSize/2,height-patchSize/2+1,grid_step))
    colorDescriptor=None
    currentOccupiedWidth=0
   
    for i,scale in enumerate(scales):    
        patchSizeAtCurrentScale=patchSize*scale
        x_grid=np.linspace(patchSizeAtCurrentScale/2,width*scale-patchSizeAtCurrentScale/2+1,Nx)
        y_grid=np.linspace(patchSizeAtCurrentScale/2,height*scale-patchSizeAtCurrentScale/2+1,Ny)
        if(scale!=1):
            resized = imutils.resize(image, width = int(image.shape[1]*scale))       
        else:
            resized=image
        colorDescriptorAtCurrentScale=calculateColorHistogramOfAllPatchsInImage(resized , [x_grid,y_grid], patchSizeAtCurrentScale, nbins=32)
        if colorDescriptor is None:
            colorDescriptor=colorDescriptorAtCurrentScale
        else:
            colorDescriptor=np.vstack((colorDescriptor,colorDescriptorAtCurrentScale))
            

    return colorDescriptor


#cv2.namedWindow("image",cv2.WINDOW_NORMAL)        
def calculateDenseSIFT3Channels(image,step_size=4,patchSize=10):
    
    kp = [cv2.KeyPoint(x, y, patchSize) for y in range(int(patchSize/2), int(image.shape[0]-patchSize/2+1), step_size)  for x in range(int(patchSize/2),int( image.shape[1]-patchSize/2+1), step_size)]    
    sift = cv2.xfeatures2d.SIFT_create()
    
    denseFeatCh1 = sift.compute(image[:,:,0], kp)[1]
    denseFeatCh1=denseFeatCh1/(np.linalg.norm(denseFeatCh1,axis=1).reshape(-1,1)+1e-7)
    
    denseFeatCh2 = sift.compute(image[:,:,1], kp)[1]
    denseFeatCh2=denseFeatCh2/(np.linalg.norm(denseFeatCh2,axis=1).reshape(-1,1)+1e-7)
     
    denseFeatCh3 = sift.compute(image[:,:,2], kp)[1]
    denseFeatCh3=denseFeatCh3/(np.linalg.norm(denseFeatCh3,axis=1).reshape(-1,1)+1e-7)
    
    denseFeat=np.hstack((np.hstack((denseFeatCh1,denseFeatCh2)),denseFeatCh3))
    
    return denseFeat.T
    
    
def extractDataForOneImage(image,grid_step=4, patchSize=10, nbins=32,scales=[1,0.75,0.5]):
    
    ### Convert to LAB colorspace
    imageInLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    ###Compute color descriptors for all patches in the image
    colorDescriptorsOfTheImage=calculateColorHistogramOfAllPatchsInImageAtDifferentScales(imageInLAB, grid_step, patchSize, nbins,scales)
    #colorDescriptorsOfTheImage=parallelColorHistogramOfAllPatchsInImageAtDifferentScales(imageInLAB, grid_step, patchSize, nbins,scales)
    
    ###Compute sift descriptors for all patches in the image
    siftDescriptorOfTheImage=calculateDenseSIFT3Channels(imageInLAB)
    
    ###Concatenate the descriptors
    descriptorsForImage=np.vstack((colorDescriptorsOfTheImage,siftDescriptorOfTheImage))

    return descriptorsForImage
    

def extractDescriptorsFromCam(camDir,featDir=None,width=None,height=None,patchSize=10,gridStep=4,nbins=32,scales=[1,0.75,0.5],level='image'):
    
    images=[os.path.join(camDir,i) for i in os.listdir(camDir) if i.endswith(('.jpg','.png','.bmp'))]
    n=len(images)
    
    if width==None or height==None:
        height,width,*_=cv2.imread(images[0]).shape
    
    Nx = len(np.arange(patchSize / 2, width - patchSize / 2 + 1, gridStep))
    Ny = len(np.arange(patchSize / 2, height - patchSize / 2 + 1, gridStep))
    
    patchDim=nbins*len(scales)*3+3*128
    patchNum=Nx*Ny
    
    if level=='image':
        feats=np.zeros((patchDim*patchNum,n))
    elif level=='patch':
        feats=np.zeros([patchDim,patchNum,n])
    
    for index, pathOfImage in enumerate(images):
        img = cv2.imread(pathOfImage)
        descriptorOneImage = extractDataForOneImage(img, grid_step=gridStep, patchSize=patchSize, nbins=nbins, scales=scales)
        feats[...,index]=descriptorOneImage.ravel(order='F') if level=='image' else descriptorOneImage
        
        if index % 10 == 0 and index!=0:
            print("Descriptor of image " + str(index) + "/" + str(len(images)) + " extracted")
            
    if featDir==None:
        return feats,Nx,Ny
    else:
        np.savez(featDir,feats=feats,nx=Nx,ny=Ny)
        print("\n Descriptors are being saved")
