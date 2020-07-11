import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
os.environ['path']=os.environ['path']+';%s'%os.path.normpath(os.path.join(os.path.dirname(__file__),'../../scripts/'))
import rarfile
import zipfile
from lomo.tools import calc_cmc,measure_time,getcwd
from zoo.cprint import cprint,fcolors,cprint_err
from functools import reduce
from collections import Iterable
import requests
from PIL import Image
from io import BytesIO
from contextlib import closing
import time
import struct

def euc_dist(X,Y=None):
    '''calc euclidean distance of X(d*m) and Y(d*n), func return 
       a dist matrix D(m*n), D[i,j] represents the distance of X[:,i] 
       and Y[:,j]'''
    # A=np.sum(X.T*(X.T),axis=1)
    # D=np.sum(Y.T*(Y.T),axis=1)
    # return A[:,None]+D[None,:]-X.T.dot(Y)*2
    A=np.sum(X.T*(X.T),axis=1,keepdims=True)
    if Y is None:
        D=A.T
    else:
        D=np.sum(Y*Y,axis=0,keepdims=True)
    return A+D-2*X.T@(X if Y is None else Y)

def euc_dist_pro(X,Y=None): #完全兼容euc_dist
    '''calc euclidean distance of X(...,d,m) and Y(...,d,n), func 
       return a dist matrix D(...,m,n), D[...,i,j] represents the 
       distance of X[...,:,i] and Y[...,:,j]'''
    XT=np.swapaxes(X,-2,-1)
    A=np.sum(XT*XT,axis=-1,keepdims=True)
    if Y is None:
        D=np.swapaxes(A,-2,-1)
    else:
        D=np.sum(Y*Y,axis=-2,keepdims=True)
    return A+D-2*np.matmul(XT,(X if Y is None else Y))

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

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        print('Create DIR %s'%os.path.normpath(dir_path))
        os.makedirs(dir_path)

def flatten(a):
    '''使用递归展开一个嵌套的多层for循环'''
    ret=[]
    for i in a:
        if isinstance(i,Iterable):
            ret.extend(flatten(i))
        else:
            ret.append(i)
    return ret

class Audio(object): #comes from https://www.jianshu.com/p/b28faf43053d
    def __init__(self):
        """播放音频"""
        from pygame import mixer
        self.pygame_mixer = mixer
        self.pygame_mixer.init()
        self.audio_bytes = None

    def play(self, audio_bytes=None):
        """传入音频文件字节码，播放音频"""
        audio_bytes = self.audio_bytes or audio_bytes
        if audio_bytes is None:
            return
        byte_obj = BytesIO()
        byte_obj.write(audio_bytes)
        byte_obj.seek(0, 0)
        self.pygame_mixer.music.load(byte_obj)
        self.pygame_mixer.music.play()
        while self.pygame_mixer.music.get_busy() == 1:
            time.sleep(0.1)
        self.pygame_mixer.music.stop()

def filetype(file): #comes from https://blog.csdn.net/mingzznet/article/details/46279753
    fileTypeList={"52617221": 'rar', "504B0304": 'zip', '89504E47': 'png', 'FFD8FF': 'jpg', 
                  '424D': 'bmp', '57415645': 'wav', '41564920': 'avi'} # 一些常见的文件类型
    def bytes2hex(bytes): # 字节码转16进制字符串
        num = len(bytes)
        hexstr = u""
        for i in range(num):
            t = u"%x" % bytes[i]
            if len(t) % 2:
                hexstr += u"0"
            hexstr += t
        return hexstr.upper()
    if isinstance(file,str) and os.path.exists(file):
        binfile = open(file, 'rb')
        ftype = 'unknown'
        for hcode in fileTypeList.keys():
            numOfBytes = int(len(hcode) / 2) # 需要读多少字节
            binfile.seek(0) # 每次读取都要回到文件头，不然会一直往后读取
            hbytes = struct.unpack_from("B"*numOfBytes, binfile.read(numOfBytes)) # 一个"B"表示一个字节
            f_hcode = bytes2hex(hbytes)
            if f_hcode == hcode:
                ftype = fileTypeList[hcode]
                break
        binfile.close()
    else: #否则必须是二进制数据
        for hcode in fileTypeList.keys():
            numOfBytes = int(len(hcode) / 2)
            hbytes = struct.unpack_from("B"*numOfBytes, file[:numOfBytes])
            f_hcode = bytes2hex(hbytes)
            if f_hcode == hcode:
                ftype = fileTypeList[hcode]
                break
    return ftype

class ProgressBar: #comes from https://www.jb51.net/article/156304.htm
    def __init__(self, title, count, total):
        self.info = "%.2f%%|%s| %s【%s】 %.2fMB/%.2fMB"
        self.title = title
        self.total = total
        self.count = count
        self.state = "正在下载"
    
    def __get_info(self):
        now=self.count/1048576
        end=self.total/1048576
        _info = self.info % (100*now/end, '█'*(int(20*(now/end)))+' '*(int(20*((end-now)/end))), \
            self.state, self.title, now, end)
        return _info
    
    def refresh(self, count):
        self.count += count
        end_str = "\r"
        if self.count >= self.total:
            end_str = '\n'
            self.state="下载完成"
        print(self.__get_info(), end=end_str)

class Crawler:
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' \
             '(KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393'}
              
    def __init__(self,url=None,save_path=None):
        self.url=url
        self.save_path=save_path
        
    def get(self,url=None,method='GET',**kwargs):
        if url is not None:
            self.url=url
        elif self.url is None:
            raise ValueError('Must give URL!')
        if kwargs.get('headers') is None:
            kwargs['headers']=self.headers
        raise_error=kwargs.get('raise_error')
        if raise_error is not None:
            del kwargs['raise_error']
        show_bar=kwargs.get('show_bar')
        if show_bar is not None:
            del kwargs['show_bar']
        if show_bar: #如果给出show_bar参数为真，则会展示下载进度条，默认不展示
            kwargs['stream']=True
            chunk_size=kwargs.get('chunk_size')
            if chunk_size is None: #在show_bar为真时，可以传递chunk_size和name参数
                chunk_size=1024*1024 # 单次请求最大值（1MB）
            else:
                del kwargs['chunk_size']
            name=kwargs.get('name')
            if name is None:
                name='未命名'
            else:
                del kwargs['name']
            self.content=None
            with closing(requests.request(method,self.url,**kwargs)) as response:
                self.content_length = float(response.headers['content-length']) # 内容体总大小
                self.content_type=response.headers['Content-Type']
                progress = ProgressBar(name, 0, total=self.content_length)
                for data in response.iter_content(chunk_size=chunk_size):
                    progress.refresh(count=len(data))
                    if self.content is None:
                        self.content=data
                    else:
                        self.content+=data
        else:
            response=requests.request(method,self.url,**kwargs)
            if raise_error: #如果给出raise_error参数为真，则在response!=200时抛出异常，默认不抛
                response.raise_for_status()
            if response.ok:
                self.content_type=response.headers['Content-Type']
                self.content_length=response.headers['Content-Length']
                self.content=response.content
                print('Get file[%s](%.2fM) from %s successfully!'%(self.content_type, \
                    float(self.content_length)/1048576,self.url))
            else:
                print('Failed(%d)!'%response.status_code)
            return self #返回爬虫自身
        
    def save(self,save_path=None):
        if save_path is not None:
            self.save_path=os.path.normpath(save_path)
            if not os.path.exists(os.path.dirname(self.save_path)):
                print('Create dir %s'%os.path.dirname(self.save_path))
                os.makedirs(os.path.dirname(self.save_path))
        elif self.save_path is None:
            self.save_path=os.path.join(getcwd(),'result') #如果没有给出保存路径，则默认保存到此函数的
                                                           #调用者所在目录下名为result.<suffix>的文件下
        suffix=filetype(self.content) #可以通过文件头标识判断，这种方式更准确
                                      #https://www.cnblogs.com/senior-engineer/p/9541719.html
        if suffix=='unknown':
            suffix=self.content_type.split('/')[-1] #获取文件后缀，用于辅助上面的文件头标识判断
            if suffix=='x-rar-compressed':
                suffix='rar'
            else:
                pass
        self.save_file_suffix=suffix
        if not self.save_path.lower().endswith(suffix.lower()): #注意这边会自动添加后缀！
            self.save_path+='.%s'%suffix
        self.save_path=os.path.normpath(self.save_path)
        with open(self.save_path,'wb') as f:
            f.write(self.content)
        print('Save to %s'%self.save_path)
        
    def show_img(self): #如果获取的是图片，可以调用此方法
        img=Image.open(BytesIO(self.content))
        img.show()
        
    def play_audio(self): #https://pythonbasics.org/python-play-sound/
        Audio().play(self.content)
        
    def __call__(self,*args,**kwargs): #这些参数将如数传递给get内部的request函数
        return self.get(*args,**kwargs)

def unzip(file_path,save_dir,file_type=None):
    if file_type is None:
        file_type=file_path.split('.')[-1]
    file_path,save_dir=os.path.normpath(file_path),os.path.normpath(save_dir)
    if file_type=='rar': #https://rarfile.readthedocs.io/faq.html#what-are-the-dependencies
        rar_file=rarfile.RarFile(file_path)
        rar_file.extractall(save_dir)
        rar_file.close()
    elif file_type=='zip':
        zip_file=zipfile.ZipFile(file_path)
        zip_file.extractall(save_dir)
        zip_file.close()
    else:
        raise ValueError('Invalid file type(%s?), Must be zip or rar!'%file_type)
    print('Extracted %s into %s'%(file_path,save_dir))

def split_dataset_trials(pids,cids,dataset,trials=10): #我将提供此处所有数据集的下载
    pids=norm_labels(pids)
    cids=norm_labels(cids)
    if dataset=='viper':
        cam_a_inds=np.where(cids==0)[0]
        cam_b_inds=np.where(cids==1)[0]
        cam_a_sorted_by_pids_inds=cam_a_inds[np.argsort(pids[cam_a_inds])]
        cam_b_sorted_by_pids_inds=cam_b_inds[np.argsort(pids[cam_b_inds])]
        for _ in range(trials):
            p=np.random.permutation(632)
            ptrain,ptest=p[:316],p[316:]
            ret={}
            ret['indsAtrain']=cam_a_sorted_by_pids_inds[ptrain]
            ret['indsBtrain']=cam_b_sorted_by_pids_inds[ptrain]
            ret['indsAtest']=cam_a_sorted_by_pids_inds[ptest]
            ret['indsBtest']=cam_b_sorted_by_pids_inds[ptest]
            ret['labelsAtrain']=pids[ret['indsAtrain']]
            ret['labelsBtrain']=pids[ret['indsBtrain']]
            ret['labelsAtest']=pids[ret['indsAtest']]
            ret['labelsBtest']=pids[ret['indsBtest']]
            ret['camlabelsAtrain']=cids[ret['indsAtrain']]
            ret['camlabelsBtrain']=cids[ret['indsBtrain']]
            ret['camlabelsAtest']=cids[ret['indsAtest']]
            ret['camlabelsBtest']=cids[ret['indsBtest']]
            yield ret
    elif dataset=='cuhk01':
        cam_a_inds=np.where(cids==0)[0]
        cam_b_inds=np.where(cids==1)[0]
        cam_a_sorted_by_pids_inds=cam_a_inds[np.argsort(pids[cam_a_inds])]
        cam_b_sorted_by_pids_inds=cam_b_inds[np.argsort(pids[cam_b_inds])]
        for _ in range(trials):
            p=np.random.permutation(485+486)
            ptrain,ptest=(p[:485]*2)[:,None],(p[485:]*2)[:,None]
            ptrain=np.hstack((ptrain,ptrain+1)).flatten()
            ptest=np.hstack((ptest,ptest+1)).flatten()
            ret={}
            ret['indsAtrain']=cam_a_sorted_by_pids_inds[ptrain]
            ret['indsBtrain']=cam_b_sorted_by_pids_inds[ptrain]
            ret['indsAtest']=cam_a_sorted_by_pids_inds[ptest]
            ret['indsBtest']=cam_b_sorted_by_pids_inds[ptest]
            ret['labelsAtrain']=pids[ret['indsAtrain']]
            ret['labelsBtrain']=pids[ret['indsBtrain']]
            ret['labelsAtest']=pids[ret['indsAtest']]
            ret['labelsBtest']=pids[ret['indsBtest']]
            ret['camlabelsAtrain']=cids[ret['indsAtrain']]
            ret['camlabelsBtrain']=cids[ret['indsBtrain']]
            ret['camlabelsAtest']=cids[ret['indsAtest']]
            ret['camlabelsBtest']=cids[ret['indsBtest']]
            yield ret
    elif dataset=='cuhk02':
        pass
        
if __name__=='__main__':
    import scipy.io as scio
    feadir=r'C:\Users\Administrator\Desktop\CAMEL-master'
    # viper_pids=list(range(632))*2
    # viper_cids=[1]*632+[2]*632
    # ret=split_dataset_trials(viper_pids,viper_cids,'viper')
    # viperfeafile=os.path.join(feadir,'viper_jstl64.mat')
    # viperfea=scio.loadmat(viperfeafile)['feature']
    # print(viperfea.shape)
    ### viper通过camel提供的matlab提取jstl特征后，直接用viperjstl分支下的split.m分割，然后执行demo_ours.m即可
    cuhk01_pids=np.broadcast_to(np.arange(485+486)[:,None],(485+486,4)).flatten().tolist()
    cuhk01_cids=[1,1,2,2]*(485+486)
    ret=split_dataset_trials(cuhk01_pids,cuhk01_cids,'cuhk01')
    cuhk01feafile=os.path.join(feadir,'cukh01_jstl64.mat')
    cuhk01feasavefile=os.path.join(feadir,'cuhk01_jstl64_save.mat')
    cuhk01fea=scio.loadmat(cuhk01feafile)['feature']
    save_data={}
    for trial_num,trial_data in enumerate(ret,1):
        save_data['trial%d'%trial_num]={'featAtrain':cuhk01fea[:,trial_data['indsAtrain']],\
            'featBtrain':cuhk01fea[:,trial_data['indsBtrain']],\
                'featAtest':cuhk01fea[:,trial_data['indsAtest']],\
                    'featBtest':cuhk01fea[:,trial_data['indsBtest']],
                    'labelsAtrain':trial_data['labelsAtrain']+1,
                    'labelsBtrain':trial_data['labelsBtrain']+1,
                    'labelsAtest':trial_data['labelsAtest']+1,
                    'labelsBtest':trial_data['labelsBtest']+1,
                    'camlabelsAtrain':trial_data['camlabelsAtrain']+1,
                    'camlabelsBtrain':trial_data['camlabelsBtrain']+1,
                    'camlabelsAtest':trial_data['camlabelsAtest']+1,
                    'camlabelsBtest':trial_data['camlabelsBtest']+1}
    scio.savemat(cuhk01feasavefile,save_data)
    ### 提取jstl特征后，由cuhk01jstl分支下的split对cuhk01_jstl64_save.mat做组织，得到CUHK01_jstl64_split.mat，再执行demo_ours.m即可