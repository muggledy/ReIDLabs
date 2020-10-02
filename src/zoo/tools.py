import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../'))
os.environ['path']=os.environ['path']+';%s'%os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../scripts/'))
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
import traceback
import pickle
import shutil
from lxml import etree
from collections import defaultdict

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
    dist=np.array(dist) #these inputs must be ndarray, otherwise got error result
    prob_identities=np.array(prob_identities)
    gal_identities=np.array(gal_identities)
    prob_id_views=np.array(prob_id_views)
    gal_id_views=np.array(gal_id_views)

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

def flatten(a,depth=-1):
    '''使用递归展开一个嵌套的多层列表，depth表示递归的最大深度，如果等于0，
       则不递归（原样输出），如果置为-1，表示无递归深度限制'''
    ret=[]
    for i in a:
        if isinstance(i,(list,tuple)) and (depth>0 or depth==-1): #isinstance(i,Iterable)
            if depth==-1:
                ret.extend(flatten(i,-1))
            elif depth>0:
                ret.extend(flatten(i,depth-1))
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

def unzip(file_path,save_dir,file_type=None):
    if file_type is None:
        file_type=file_path.split('.')[-1]
    file_path,save_dir=os.path.normpath(file_path),os.path.normpath(save_dir)
    if file_type=='rar': #https://rarfile.readthedocs.io/faq.html#what-are-the-dependencies
        rar_file=rarfile.RarFile(file_path)
        rar_file.extractall(save_dir) #在其他机子测试出现错误rarfile.RarCannotExec: Cannot find working tool
                                      #实际是第三方库rarfile版本不对，版本见/environment.yaml
        rar_file.close()
    elif file_type=='zip':
        zip_file=zipfile.ZipFile(file_path)
        zip_file.extractall(save_dir)
        zip_file.close()
    else:
        raise ValueError('Invalid file type(%s?), Must be zip or rar!'%file_type)
    print('Extracted %s into %s'%(file_path,save_dir))

class Crawler:
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' \
             '(KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393'}
              
    def __init__(self,url=None,save_path=None):
        self.url=url
        self.save_path=save_path
        self.tempf=os.path.join(getcwd(),'crawler_temp.pickle')
        self.content=None
        self.ok=False
        
    def get(self,url=None,method='GET',**kwargs): #最初设计用于下载诸如图片、音乐、压缩包等（二进制）文件，
                                                  #若用于网页则会报错，譬如响应头没有content-length等属性
                                                  #目前已捕捉异常，不会抛出影响其他程序执行
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
        
        #这里可以往kwargs中添加代理，使用关键字参数proxies，譬如：proxies = {'http':'http://10.10.10.10:8765','https':'https://10.10.10.10:8765'}
        #https://www.kuaidaili.com/free/
        #很多时候，返回错误requests.exceptions.ConnectionError: HTTPSConnectionPool(host='1drv.ws', port=443): Max retries exceeded with url
        #就是这个原因，使用合适的代理就可以解决
        use_proxy=kwargs.get('use_proxy')
        if use_proxy is not None:
            del kwargs['use_proxy']
        if use_proxy:
            proxies_dict=get_proxy_kusidaili(pages=3,read_from_local=True)
            proxies_t={}
            for _type in proxies_dict:
                proxies_t[_type]=np.random.choice(proxies_dict[_type])
            kwargs['proxies']=proxies_t
            print('use proxy:%s'%(','.join(list(kwargs['proxies'].values()))))

        try:
            if show_bar: #如果给出show_bar参数为真，则会展示下载进度条，默认不展示，在此逻辑下支持断点续传，由缓存机制支持
                         #https://www.cnblogs.com/skying555/p/6218384.html
                kwargs['stream']=True
                chunk_size=kwargs.get('chunk_size')
                if chunk_size is None: #只有在show_bar为真时，才可以传递chunk_size和name参数
                    chunk_size=1024*1024 # 单次请求最大值设为1MB
                else:
                    del kwargs['chunk_size']
                name=kwargs.get('name')
                if name is None:
                    name='未命名'
                else:
                    del kwargs['name']
                
                #缓存机制
                if not os.path.exists(self.tempf):
                    with open(self.tempf,'wb') as crawler_temp_f:
                        pickle.dump({},crawler_temp_f)
                with open(self.tempf,'rb') as crawler_temp_f:
                    logs_dict=pickle.load(crawler_temp_f)
                if logs_dict.get(self.url)==None:
                    logs_dict[self.url]=[os.path.join(os.path.dirname(self.tempf),'%s.temp'% \
                        (time.strftime("%Y-%m-%d %H.%M.%S",time.localtime(time.time())))),-1] #最好再加一个是否下载完成的状态码
                    with open(self.tempf,'wb') as crawler_temp_f:
                        pickle.dump(logs_dict,crawler_temp_f)
                if os.path.exists(logs_dict[self.url][0]):
                    local_file_size=os.stat(logs_dict[self.url][0]).st_size
                    kwargs['headers']['Range']='bytes=%d-'%local_file_size
                else:
                    local_file_size=0
                if local_file_size==logs_dict[self.url][1]:
                    print('已下载完毕')
                    self.ok=True
                    return self
                
                with closing(requests.request(method,self.url,**kwargs)) as response:
                    if raise_error:
                        response.raise_for_status()
                    self.ok=response.ok and (response.status_code in [200,206]) #状态码206表示断点续传
                    if self.ok:
                        self.content_length = float(response.headers['content-length']) # 内容体总大小
                        self.content_type=response.headers['Content-Type']
                        progress = ProgressBar(name, 0, total=self.content_length+local_file_size)
                        progress.refresh(count=local_file_size)
                        # 缓存机制
                        if logs_dict[self.url][1]==-1:
                            logs_dict[self.url][1]=self.content_length
                            with open(self.tempf,'wb') as crawler_temp_f:
                                pickle.dump(logs_dict,crawler_temp_f)
                        with open(logs_dict[self.url][0],'ab') as local_temp_f:
                            for data in response.iter_content(chunk_size=chunk_size):
                                progress.refresh(count=len(data))
                                # if self.content is None: #我不想占用内存，所以当使用show_bar方式下载，在任何时候都可以
                                                           #将self.content置为None，因为数据是保存在本地临时文件中
                                #     self.content=data
                                # else:
                                #     self.content+=data
                                local_temp_f.write(data)
                                local_temp_f.flush()
                    else:
                        print('Crawler Failed(%d) from %s!'%(response.status_code,self.url))
            else: #可以下载小文件，该逻辑块不支持断点续传，也不show_bar。但是不会因为不存在content-length就报错
                print('Downloading from %s...'%self.url)
                response=requests.request(method,self.url,**kwargs)
                if raise_error: #如果给出raise_error参数为真，则在response!=200时抛出异常，默认不抛
                    response.raise_for_status()
                self.ok=response.ok and (response.status_code==200)
                if self.ok:
                    self.content_type=response.headers.get('Content-Type')
                    self.content_length=response.headers.get('Content-Length')
                    self.content=response.content
                    print('Get response%s%s successfully!'%('[%s]'%self.content_type if self.content_type is not None else '', \
                        '(%.2fM)'%(float(self.content_length)/1048576) if self.content_length is not None else ''))
                else:
                    print('Crawler Failed(%d)!'%response.status_code)
        # except requests.HTTPError:
        #     raise
        # except Exception as e:
        #     self.ok=False
        #     # print('***',type(e),e,'***')
        #     traceback.print_exc()
        except:
            raise
        return self #返回爬虫自身
        
    def save(self,save_path=None,auto_detect_suffix=True):
        if self.ok:
            if save_path is not None:
                self.save_path=os.path.normpath(save_path)
                if not os.path.exists(os.path.dirname(self.save_path)):
                    print('Create dir %s'%os.path.dirname(self.save_path))
                    os.makedirs(os.path.dirname(self.save_path))
            elif self.save_path is None:
                self.save_path=os.path.join(getcwd(),'result') #如果没有给出保存路径，则默认保存到此函数的
                                                               #调用者所在目录下名为result.<suffix>的文件下
            from_local=False
            if self.content is None: #如果self.content为空，则表示使用的是show_bar方式下载，文件已经保存在本地临时文件中
                from_local=True
                self.content=self.read_from_temp()
            if auto_detect_suffix: #如果给出的save_path不包含文件后缀，也就是需要自动检测后缀标志为真时
                # print('auto detect suffix of downloaded binary file')
                suffix=filetype(self.content) #可以通过文件头标识判断，这种方式更准确
                                          #https://www.cnblogs.com/senior-engineer/p/9541719.html
                # if suffix=='unknown':
                #     suffix=self.content_type.split('/')[-1] #从响应头中获取文件后缀，用于辅助上面的文件头标识判断法
                #     if suffix=='x-rar-compressed':
                #         suffix='rar'
                #     else:
                #         pass
                self.save_file_suffix=suffix
                if not self.save_path.lower().endswith(suffix.lower()): #注意这边会自动添加后缀！
                    self.save_path+='.%s'%suffix
                self.save_path=os.path.normpath(self.save_path)
            else: #若save_path已经包含后缀
                suffix=os.path.splitext(self.save_path)[1].split('.')[-1].lower()
                self.save_file_suffix=suffix
            # if from_local:
            #     rename=os.path.join(os.path.dirname(logs_dict[self.url][0]),os.path.basename(self.save_path))
            #     os.rename(logs_dict[self.url][0],rename)
            #     if not os.path.normpath(rename)==os.path.normpath(self.save_path):
            #         shutil.move(rename,os.path.dirname(self.save_path))
            # else:
            #     with open(self.save_path,'wb') as f:
            #         f.write(self.content)
            with open(self.save_path,'wb') as f:
                f.write(self.content)
            print('Saved to %s'%self.save_path)
            # del logs_dict[self.url]
            # with open(self.tempf,'wb') as crawler_temp_f:
            #     pickle.dump(logs_dict,crawler_temp_f)
            if from_local: #当下载大文件，建议使用show_bar方式，不会长期占用大量内存
                self.content=None
        else:
            print('Save Nothing(Crawler Failed)!')
        
    def show_img(self): #如果获取的是图片，可以调用此方法
        if self.ok:
            from_local=False
            if self.content is None:
                from_local=True
                self.content=self.read_from_temp()
            img=Image.open(BytesIO(self.content))
            if from_local:
                self.content=None
            img.show()
        else:
            print('Show Nothing(Crawler Failed)!')
        
    def play_audio(self): #https://pythonbasics.org/python-play-sound/
        if self.ok:
            from_local=False
            if self.content is None:
                from_local=True
                self.content=self.read_from_temp()
            Audio().play(self.content)
            if from_local:
                self.content=None
        else:
            print('Play Nothing(Crawler Failed)!')

    def read_from_temp(self,url=None):
        if url==None:
            url=self.url
        with open(self.tempf,'rb') as crawler_temp_f:
            logs_dict=pickle.load(crawler_temp_f)
            with open(logs_dict[url][0],'rb') as local_temp_f:
                content=local_temp_f.read()
        return content

    def clear_temp(self,url=None,check_finish=True):
        if url==None:
            url=self.url
        if url==None:
            raise ValueError('(clear_temp)Search URL can\'t be None!')
        if os.path.exists(self.tempf):
            with open(self.tempf,'rb') as crawler_temp_f:
                logs_dict=pickle.load(crawler_temp_f)
            if logs_dict.get(url)==None:
                print('(clear_temp)No this record or the temp file has been cleared!')
            else:
                delete_flag=True
                local_temp_f=logs_dict[url][0]
                if os.path.exists(local_temp_f):
                    if check_finish:
                        local_file_size=os.stat(local_temp_f).st_size
                        if local_file_size!=logs_dict[url][1]:
                            delete_flag=False
                    if delete_flag:
                        os.remove(local_temp_f)
                if delete_flag:
                    print('(clear_temp)Delete record and temp file %s'%local_temp_f)
                    del logs_dict[url]
                    if not logs_dict: #如果记录为空的话，连同记录文件也删除
                        if os.path.exists(self.tempf):
                            os.remove(self.tempf)
                    else:
                        with open(self.tempf,'wb') as crawler_temp_f:
                            pickle.dump(logs_dict,crawler_temp_f)
                else:
                    print('(clear_temp)Because temp file isn\'t finished(CHECKED), NO delete!')
        else:
            raise ValueError('(clear_temp)No temp recorder file(%s)!'%self.tempf)

    def unzip(self,save_dir=None,delete=False): #注意unzip之前必须先save压缩文件！
        if self.ok:
            if not os.path.exists(self.save_path):
                raise ValueError('Must save the rar/zip file firstly, then do unzip!')
            if save_dir is not None:
                save_dir=os.path.normpath(save_dir)
            else: #默认解压到save_path所在目录
                save_dir=os.path.dirname(self.save_path)
            if not os.path.exists(save_dir):
                print('Create dir %s'%save_dir)
                os.makedirs(save_dir)
            unzip(self.save_path,save_dir,self.save_file_suffix)
            if delete:
                print('(unzip)Delete origin zip/rar file %s'%self.save_path)
                os.remove(self.save_path)
        else:
            print('Extract Nothing(Crawler Failed)!')
        
    def __call__(self,*args,**kwargs): #这些参数将如数传递给get内部的request函数
        return self.get(*args,**kwargs)

#谷歌云盘：pip install google-cloud-storage

def split_dataset_trials(pids,cids,dataset,trials=10): #通过/images/download_dataset.py下载这些数据集，
                                                       #注意是根据其在Windows下的读取顺序设计的
    pids=norm_labels(pids)
    cids=norm_labels(cids)
    if dataset in ['viper','cuhk01','prid_single']:
        if dataset=='viper' and (len(pids)!=1264 or len(cids)!=1264 or len(set(pids))!=632 or len(set(cids))!=2): #基本检查
            raise ValueError('Invalid VIPeR dataset!')
        elif dataset=='cuhk01' and (len(pids)!=3884 or len(cids)!=3884 or len(set(pids))!=971 or len(set(cids))!=2):
            raise ValueError('Invalid CUHK01 dataset!')
        elif dataset=='prid_single' and (len(pids)!=(200+749) or len(cids)!=(200+749) or len(set(pids))!=749 or len(set(cids))!=2):
            raise ValueError('Invalid PRID(single) dataset!')
        cam_a_inds=np.where(cids==0)[0]
        cam_b_inds=np.where(cids==1)[0]
        if dataset=='prid_single' and (len(cam_a_inds)>len(cam_b_inds)):
            cam_a_inds,cam_b_inds=cam_b_inds,cam_a_inds #不同于viper既可以摄像头A作probe，也可以摄像头B作probe，prid只能是小的那个作probe
        
        cam_a_sorted_by_pids_inds=cam_a_inds[np.argsort(pids[cam_a_inds])]
        cam_b_sorted_by_pids_inds=cam_b_inds[np.argsort(pids[cam_b_inds])]
        for _ in range(trials):
            if dataset=='viper':
                p=np.random.permutation(632)
                ptrain,ptest=p[:316],p[316:] #VIPeR有632个行人，每个行人在总共两个摄像头下各自有一张图像，
                                             #挑出316个行人数据作为训练集，剩下316个作为测试集
            elif dataset=='cuhk01':
                p=np.random.permutation(485+486)
                ptrain,ptest=(p[:485]*2)[:,None],(p[485:]*2)[:,None] #CUHK01有971个行人，每个行人在总计两个摄像头下各自有两张
                                                                     #图像，挑出485个行人数据作为训练集，剩下486个作为测试集
                ptrain=np.hstack((ptrain,ptrain+1)).flatten()
                ptest=np.hstack((ptest,ptest+1)).flatten()
            elif dataset=='prid_single': #prid(single)数据集不同于viper或cuhk01，虽然也是两个摄像头，但是不同摄像头下行人图像数量不同，
                                         #摄像头A（作为probe）下有385个行人，摄像头B（作为gallery）下有749个行人，只有前200个行人同时出
                                         #现在两个交叉摄像头下，摄像头A和摄像头B下的图像总数分别为385和749，也就是说，每个行人在某一摄像
                                         #头下最多有且只有一张图像。训练时，从交叉摄像头下共有的200个行人中挑出100个行人总计两百张图像作
                                         #为训练集，虽然摄像头A下有385个行人，但是其中185个从未出现在摄像头B下，在测试中也不会使用（训练
                                         #时也用不到），即仅将剩下的100个行人作为probe，而摄像头B下剩余的649个行人则全部作为gallery
                p=np.random.permutation(200)
                ptrain,ptest=p[:100],p[100:]
            
            ret={}
            ret['indsAtrain']=cam_a_sorted_by_pids_inds[ptrain]
            ret['indsBtrain']=cam_b_sorted_by_pids_inds[ptrain]
            ret['indsAtest']=cam_a_sorted_by_pids_inds[ptest]
            if dataset=='prid_single':
                ret['indsBtest']=cam_b_sorted_by_pids_inds[ptest.tolist()+list(range(200,749))]
            else:
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
    elif dataset=='':
        pass

def get_proxy_kusidaili(base_url='https://www.kuaidaili.com/free/inha/',pages=20,read_from_local=False): #comes from https://www.77169.net/html/262105.html
    def _f(url):
        header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
        response = requests.get(url=url, headers=header)
        selector = etree.HTML(response.content.decode())
        d=defaultdict(list)
        for i in range(1,16):
            _ip = selector.xpath('//*[@id="list"]/table/tbody/tr[%d]/td[1]/text()' %(i))[0]
            _port = selector.xpath('//*[@id="list"]/table/tbody/tr[%d]/td[2]/text()' %(i))[0]
            _type = selector.xpath('//*[@id="list"]/table/tbody/tr[%d]/td[4]/text()'%(i))[0].lower()
            d[_type].append('%s://%s:%s'%(_type,_ip,_port))
        return d
    
    local_save_file=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../data/kuaidaili.pickle'))
    if read_from_local and os.path.exists(local_save_file):
        print('get proxies from local %s'%local_save_file)
        with open(local_save_file,'rb') as f:
            ret=pickle.load(f)
        return ret
    tdicts=defaultdict(list)
    print('get proxies from the Internet(%s)...'%base_url)
    for i in range(1,pages+1):
        print('page %d'%i,end='\r')
        for k,v in _f('%s%s'%(base_url,str(i))).items():
            tdicts[k].extend(v)
        time.sleep(1.5+np.random.rand()*3.5)
    with open(local_save_file,'wb') as f:
        print('save proxies to local %s'%local_save_file)
        pickle.dump(tdicts,f)
    return tdicts

def gauss_blur(img,sigma): #doing gaussian blur with two-direction 1-D kernel filter
    def get_gauss_kernel(sigma,dim=2):
        ksize=int(np.floor(sigma*6)/2)*2+1
        k_1D=np.arange(ksize)-ksize//2
        k_1D=np.exp(-k_1D**2/(2*sigma**2))
        k_1D=k_1D/np.sum(k_1D)
        if dim==1:
            return k_1D
        elif dim==2:
            return k_1D[:,None].dot(k_1D.reshape(1,-1))
    row_filter=get_gauss_kernel(sigma,1)
    t=cv2.filter2D(img,-1,row_filter[...,None])
    return cv2.filter2D(t,-1,row_filter.reshape(1,-1))

if __name__=='__main__':
    url='https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBZ204d3BjSVhDanNnNHRuV2VyWDNxWk9BM0JBcUE=.jpg'
    # url='http://ting6.yymp3.net:82/new10/gaojin/12.mp3'
    crawler=Crawler(url)
    crawler(show_bar=False,use_proxy=False)
    crawler.show_img()
    # crawler.play_audio()
    # crawler.clear_temp()
    # get_proxy_kusidaili(pages=3,read_from_local=True)