class gog_lfparam:
    def __init__(self,lf_type=None,lf_bins=None):
        self.lf_type=lf_type
        self.lf_bins=lf_bins
    @property
    def lf_type(self):
        return self.__lf_type
    @lf_type.setter
    def lf_type(self,val):
        '''define the base pixel feature components, specifically we can 
           extract color information from different color spaces, 
           including RGB(lf_type=0), Lab(lf_type=1),HSV(lf_type=2) and 
           nRnG(lf_type=3)'''
        if not val in [None,0,1,2,3]:
            raise ValueError('lfparam.lf_type\'s value must be 0,1,2,3 or None!')
        self.__lf_type=0 if val==None else val
        self.reset()
    @property
    def lf_bins(self):
        return self.__lf_bins
    @lf_bins.setter
    def lf_bins(self,val):
        '''number of orientation bins'''
        self.__lf_bins=4 if val==None else val
        self.reset()
    def __getattr__(self,name): #if no obj.name, don't raise AttributeError, return None
        return None
    def reset(self):
        if self.lf_type!=None and self.lf_bins!=None:
            self.num_element=1+self.lf_bins+(3 if self.lf_type in [0,1,2] else 2)
            self.lf_name='yMtheta%d%s'%(self.lf_bins,['RGB','Lab','HSV','nRnG'][self.lf_type])

class gog_param: #there is no need to be so complex factly, a dict object is enough
    def __init__(self,lf_type=None,lf_bins=None,p=None,k=None,epsilon0=None,ifweight=None,G=None):
        '''set parameters for GoG descriptors'''
        self.__lfparam=gog_lfparam(lf_type,lf_bins)
        self.p=p
        self.k=k
        self.epsilon0=epsilon0
        self.ifweight=ifweight
        self.G=G
    @property
    def lfparam(self):
        return self.__lfparam
    
    @property
    def p(self):
        return self.__p
    @p.setter
    def p(self,val):
        '''intervals of patch extraction, default: 2'''
        self.__p=2 if val==None else int(val)
    @property
    def k(self):
        return self.__k
    @k.setter
    def k(self,val):
        '''size of patch (k*k pixles), default: 5'''
        self.__k=5 if val==None else int(val)
    @property
    def epsilon0(self):
        return self.__epsilon0
    @epsilon0.setter
    def epsilon0(self,val):
        '''regularization paramter of covariance, default: 0.001'''
        if val!=None and (val<=0 or val>=0.1):
            raise ValueError('epsilon0 must be a very small positive number or be None!')
        self.__epsilon0=0.001 if val==None else val
    @property
    def ifweight(self):
        return self.__ifweight
    @ifweight.setter
    def ifweight(self,val):
        '''if use patch weight, False: not use, True: use, default: False'''
        if not val in [None,True,False]:
            raise ValueError('ifweight must be True,False or None!')
        self.__ifweight=False if val==None else val
    @property
    def G(self):
        return self.__G
    @G.setter
    def G(self,val):
        '''number of horizontal strips, default: 7'''
        self.__G=7 if val==None else int(val)
    
    @property
    def d(self):
        '''dimension of pixel features'''
        return self.lfparam.num_element
    @property
    def m(self):
        '''dimension of patch Gaussian vector'''
        return (self.d**2+self.d*3)//2+1
    @property
    def dimlocal(self):
        '''dimension of region Gaussian vector'''
        return (self.m**2+self.m*3)//2+1
    @property
    def dimension(self):
        '''dimension of feature vector'''
        return self.dimlocal*self.G
    @property
    def name(self):
        return 'GoG%s'%self.lfparam.lf_name

def get_default_parameter(lf_type=None):
    return gog_param(lf_type=lf_type)

if __name__=='__main__':
    a=gog_param(lf_type=3)
    a.lfparam.lf_bins=8
    print(a.lfparam.lf_name,a.lfparam.num_element)
    