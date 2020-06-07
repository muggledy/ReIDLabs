import numpy as np
import cv2
from .lomo.tools import calc_cmc
from .cprint import cprint,fcolors
from functools import reduce

def euc_dist(X,Y):
    '''calc euclidean distance of X(d*m) and Y(d*n), func return 
       a dist matrix D(m*n), D[i,j] represents the distance of X[:,i] 
       and Y[:,j]'''
    A=np.sum(X.T*(X.T),axis=1)
    D=np.sum(Y.T*(Y.T),axis=1)
    return A[:,None]+D[None,:]-X.T.dot(Y)*2

def norm_labels(labels):
    '''make category labels grow from 0 one by one, for example: 
       [1 7 2 2 2 3 5 1] -> [0 4 1 1 1 2 3 0]'''
    labels=np.array(labels)
    t=np.arange(np.max(labels)+1)
    if len(t)==len(labels) and np.sum(labels-t)==0:
        return labels
    ret=labels.copy()
    s=np.sort(np.unique(labels))
    for i,e in enumerate(s):
        ret[np.where(labels==e)]=i
    return ret

def range2(start,end,step):
    cur=start
    while cur<=end:
        yield cur
        cur+=step

def gen_program(n):
    program='def gen_01(step=0.1):\n    s1=0\n'
    for i in range(1,n):
        program+=('    '*i+'for i%d in range2(0,1-s%d,step):\n'%(i,i)
            +'    '*(i+1)+'s%d=s%d+i%d\n'%(i+1,i,i))
    program+=('    '*n+'yield (%s1-s%d)'%( \
            str(''.join(['i%d,'%i for i in range(1,n)])),n))
    exec(program,globals())

def seek_good_coeffi(dists,pLabels,gLabels,numRank=100,step=0.01):
    '''dists' horizonital axis indicates probe and ..., coefficients 
       for different dist matrix are trade-off, range from (0,1)'''
    n=len(dists)
    gen_program(n)
    max_rank1=0
    max_coeffi=None
    for i,e in enumerate(gen_01(step)):
        if i%100==0:
            print(i,end='\r')
        rank1=calc_cmc(reduce(lambda x,y:x+y, \
            [xe*dists[xi] for xi,xe in enumerate(e)]), \
                pLabels,gLabels,numRank)[0]
        if rank1>max_rank1:
            max_rank1=rank1
            max_coeffi=e
    return max_coeffi,max_rank1

def create_ellipse_mask(shape):
    '''create an ellipse mask(one channel) with arg shape(h,w), h//2 
       and w//2 represent the length of ellipse's two axises'''
    h,w=shape
    ret=np.zeros(shape)
    cv2.ellipse(ret,(w//2,h//2),(w//2,h//2),0,0,360,(255,255,255),-1)
    return ret.astype('uint8')

def sym_matric(x):
    '''create a randn symmetric matric defaultly with shape(x,x). Or 
       you can give half data(x, a list) with length (1+n)*n/2, also 
       you can pass a square matric, we will copy its upper triangle'''
    if isinstance(x,int):
        M=np.random.randn(x,x)
    elif isinstance(x,list):
        t=np.sqrt(len(x)*2)
        n=int(np.floor(t))
        if n*(n+1)/2!=len(x):
            raise ValueError \
                ('x\'s length must be subjected to (1+n)*n/2=len(x)!')
        else:
            M=np.zeros((n,n))
            indx,indy=np.triu_indices(n)
            M[indx,indy]=x
    elif isinstance(x,np.ndarray) and len(x.shape)==2 \
                  and x.shape[0]==x.shape[1]:
        M=x
    else:
        raise ValueError('x must be square matric!')
    Mt=np.triu(M)
    Mt+=Mt.T-np.diag(Mt.diagonal())
    return Mt

def norm1_dist(X,Y):
    '''dist[i,j]=||x[:,i]-y[:,j]||_1'''
    return np.sum(np.abs(X[:,:,None]-Y[:,None,:]),axis=0)

def cosine_dist(X,Y):
    '''calc cosine dist of X[:,i] and Y[:,j], i.e. dist[i,j], with range 
       [0,2], dist value 0 means the most similar and dist value 2 means 
       the most dissimilar'''
    similarity=X.T.dot(Y)/(np.sqrt(np.sum(X**2,axis=0))[:,None]* \
               np.sqrt(np.sum(Y**2,axis=0)))
    return 1-similarity

def construct_I(size,d):
    '''return a block matrix where each block is an identity matrix(d,d), 
       so the return's size may be (size[0]*d,size[1]*d)'''
    I=np.zeros((size[0]*d,size[1]*d))
    for i in range(size[0]): #do you have any other good method? I don't like for-loop
        for j in range(size[1]):
            I[i*d:(i+1)*d,j*d:(j+1)*d]=np.eye(d)
    return I

def norm_labels_simultaneously(*labels):
    '''e.g. work for calc_cmc, in this case we want to normalize labels 
       for probe and gallery simultaneously, it maybe false if you norm 
       separately'''
    t=np.cumsum(np.array([len(i) for i in labels][:-1]))
    ret=norm_labels(np.hstack(labels))
    return np.split(ret,t)

def evaluate_with_index(sorted_similarity_index, right_result_index, junk_result_index=None):
    """calculate cmc curve and Average Precision for a single query with index
    :param sorted_similarity_index: index of all returned items. typically get with
        function `np.argsort(similarity)`
    :param right_result_index: index of right items. such as items in gallery
        that have the same id but different camera with query
    :param junk_result_index: index of junk items. such as items in gallery
        that have the same camera and id with query
    :return: single cmc, Average Precision
    :note: this demo comes from 
    https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/
    """
    # initial a numpy array to store the AccK(like [0, 0, 0, 1, 1, ...,1]).
    cmc = np.zeros(len(sorted_similarity_index))
    ap = 0.0

    if len(right_result_index) == 0:
        cmc[0] = -1
        return cmc, ap
    if junk_result_index is not None:
        # remove junk_index
        # all junk_result_index in sorted_similarity_index has been removed.
        # for example:
        # (sorted_similarity_index, junk_result_index)
        # ([3, 2, 0, 1, 4],         [0, 1])             -> [3, 2, 4]
        need_remove_mask = np.in1d(sorted_similarity_index, junk_result_index, invert=True)
        sorted_similarity_index = sorted_similarity_index[need_remove_mask]

    mask = np.in1d(sorted_similarity_index, right_result_index)
    right_index_location = np.argwhere(mask == True).flatten()

    # [0,0,0,...0, 1,1,1,...,1]
    #              |
    #  right answer first appearance
    cmc[right_index_location[0]:] = 1

    for i in range(len(right_result_index)):
        precision = float(i + 1) / (right_index_location[i] + 1)
        if right_index_location[i] != 0:
            # last rank precision, not last match precision
            old_precision = float(i) / (right_index_location[i])
        else:
            old_precision = 1.0
        ap = ap + (1.0 / len(right_result_index)) * (old_precision + precision) / 2

    return cmc, ap

def get_right_and_junk_index(query_label, gallery_labels, query_camera_label=None, \
                             gallery_camera_labels=None):
    '''same origin with func evaluate_with_index'''
    same_label_index = np.argwhere(gallery_labels == query_label)
    if (query_camera_label is not None) and (gallery_camera_labels is not None):
        same_camera_label_index = np.argwhere(gallery_camera_labels == query_camera_label)
        # the index of mis-detected images, which contain the body parts.
        junk_index1 = np.argwhere(gallery_labels == -1)
        # find index that are both in query_index and camera_index
        # the index of the images, which are of the same identity in the same cameras.
        junk_index2 = np.intersect1d(same_label_index, same_camera_label_index)
        junk_index = np.append(junk_index2, junk_index1)

        # find index that in query_index but not in camera_index
        # which means the same lable but different camera
        right_index = np.setdiff1d(same_label_index, \
                      same_camera_label_index, assume_unique=True)
        return right_index, junk_index
    else:
        return same_label_index, None

def calc_cmc_map(dist,prob_identities,gal_identities,prob_id_views,gal_id_views):
    '''dist's vertical axis indicates gal and horizontal prob. Same origin with 
       func evaluate_with_index'''
    gal_n,prob_n=dist.shape
    similarity=-dist
    total_cmc = np.zeros(gal_n)
    total_average_precision = 0.0
    for i in range(prob_n):
        cmc, ap = evaluate_with_index(np.argsort(similarity[:,i])[::-1],
                  *get_right_and_junk_index(prob_identities[i], gal_identities, \
                  prob_id_views[i], gal_id_views))

        if cmc[0] == -1:
            continue
        total_cmc += cmc
        total_average_precision += ap

    return total_cmc / prob_n, total_average_precision / prob_n

def print_cmc(cmc,s=[1,5,10,20,100],color=False):
    t=['Rank-%d:%.2f%%'%(i,cmc[i-1]*100) for i in s]
    flag=True if len(s)<len(fcolors) else False
    if color:
        for i,e in enumerate(t):
            f,b=e.split(':')
            cprint('%s:'%f,fcolor='white',end='')
            cprint(b,fcolor=(fcolors[i] if flag else 'blue'),end=' ')
        print()
    else:
        print(' '.join(t))

class CellMatrix2D:
    '''cell=CellMatrix2D([2,3,4],[3,4,5])
       print(cell[(0,0)])'''
    def __init__(self,xticks,yticks,dtype=None):
        self.data=np.zeros((sum(xticks),sum(yticks)),dtype=np.float32 if dtype==None else dtype)
        self.xticks,self.yticks=xticks,yticks
        self.xcum=np.cumsum([0,]+xticks)
        self.ycum=np.cumsum([0,]+yticks)
        
    def __setitem__(self,pos,data):
        s1,s2=self.get_slice(pos)
        t=(self.xticks[pos[0]],self.yticks[pos[1]])
        if data.shape!=t:
            raise ValueError('data\'s shape must be %s!'%str(t))
        self.data[s1,s2]=data
        
    def __getitem__(self,pos):
        s1,s2=self.get_slice(pos)
        return self.data[s1,s2]
        
    @property
    def info(self):
        if self.__info==None:
            t=''
            for i in self.xticks:
                t_=''
                for j in self.yticks:
                    t_+=('(%s,%s) '%(format(i, '^3d'),format(j, '^3d')))
                else:
                    t_+='\n'
                t+=t_
            self.__info=t
        return self.__info
        
    def __getattr__(self,name):
        return None
        
    def get_slice(self,pos):
        x,y=pos
        tx=(len(self.xcum)-2)
        ty=(len(self.ycum)-2)
        if x>=0 and x<=tx and y>=0 and y<=ty:
            return slice(self.xcum[x],self.xcum[x+1]),slice(self.ycum[y],self.ycum[y+1])
        else:
            raise ValueError('Invalid pos! x must be in range[0,%d], y must be in range[0,%d]!'%(tx,ty))
