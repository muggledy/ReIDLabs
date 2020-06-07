import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .tools import norm_labels
from .lomo.tools import getcwd
from matplotlib.path import Path
from matplotlib import patches

def plot_dataset(X,identity_labels,camera_labels=None,method='PCA',dim=2):
    '''X is the dataset which X[:,i] represents one sample, method 
       can be 'PCA' or 'LDA', dim can be 2 or 3. Different camera has different 
       marker, different identity has different color in the plotting'''
    identity_labels=norm_labels(identity_labels)
    if method=='PCA':
        pca=PCA(n_components=dim)
        X=pca.fit_transform(X.T)
    elif method=='LDA':
        lda=LinearDiscriminantAnalysis(n_components=dim)
        X=lda.fit_transform(X.T,identity_labels)
    X=X.T
    if camera_labels==None:
        camera_labels=np.zeros(len(identity_labels))
    else:
        camera_labels=norm_labels(camera_labels)
    camera_markers=['+','^','.','*','s','D','x','1'] #8 preset camera tags, so 8 is 
                                                     #the maximum num of cameras
    identity_colors=cm.gist_rainbow(Normalize(0,1)(np.linspace(0,1, \
                                                   np.max(identity_labels)+1))) 
                                                     #there are not thousands of colors 
                                                     #for human eyes to distinguish
    cam_num=int(np.max(camera_labels))+1
    if dim==2:
        ax=plt.gca()
    if dim==3:
        ax=Axes3D(plt.gcf())
    for i in range(cam_num):
        t=np.where(camera_labels==i)
        cam_data=X[:,t]
        ax.scatter(*cam_data,marker=camera_markers[i],c= \
                                         identity_colors[identity_labels[t]])
    plt.show()

def plot_sal_graph_W(W,xids=None,yids=None,sal=True,roi_mask=None):
    '''xids represents W's 0 axis and yids represents W's 1 axis
       , we will give a darker color to the larger weighted 
       position and salience the one-to-one correspondent pos with 
       red square on your interest'''
    plt.imshow(W,cmap='Blues')
    plt.colorbar(shrink=0.5)
    ax = plt.gca()
    if sal:
        if xids is None and yids is None:
            h,w=W.shape
            xids=np.arange(h)
            yids=np.arange(w)
        else:
            xids=np.squeeze(xids)
            yids=np.squeeze(yids)
        match=xids.reshape(-1,1)==yids
        if roi_mask is not None:
            match=(match&roi_mask.astype('bool')).astype('bool')
        x,y=np.where(match==True)
        nrects=len(x)
        points = np.zeros((nrects*5,2))
        codes = np.ones(nrects*5, int) * Path.LINETO
        bottom=x-0.5
        top=bottom+1
        left=y-0.5
        right=left+1
        codes[::5] = Path.MOVETO
        codes[4::5] = Path.CLOSEPOLY
        points[::5,0] = left
        points[::5,1] = bottom
        points[1::5,0] = left
        points[1::5,1] = top
        points[2::5,0] = right
        points[2::5,1] = top
        points[3::5,0] = right
        points[3::5,1] = bottom
        path = Path(points,codes)
        pathpatch = patches.PathPatch(path,color='red',fill=False)
        ax.add_patch(pathpatch)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    '''demo of plot_dataset
    viper=np.load(os.path.normpath(os.path.join(getcwd(),'../../data/lomo_features_viper.npz')))
    viper=np.vstack((viper['probe'],viper['gallery']))
    n=viper.shape[0]//2
    plot_dataset(viper.T,list(range(n))*2,[0]*n+[1]*n,method='PCA',dim=2)
    '''
    from sklearn import datasets
    iris=datasets.load_iris()
    n=iris.data.shape[0]
    m=int(n/4)
    plot_dataset(iris.data.T,[0]*m+[1]*m+[2]*m+[3]*(n-3*m),[0]*(n//2)+[1]*(n-n//2),dim=2)
    