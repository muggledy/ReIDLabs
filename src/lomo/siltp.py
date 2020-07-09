'''
2019/11/21
'''

import numpy as np

def calc_siltp2(imgs,N=4,R=3,tau=0.3):
    '''the same as source code, note that N must be 4 or 8'''
    h,w,n=imgs.shape
    I0=np.zeros((h+2*R,w+2*R,n),'uint8')
    I0[R:-R,R:-R,:]=imgs
    
    I0[:R,:,:]=np.broadcast_to(I0[R,:,:],(R,*(I0.shape[1:])))
    I0[-R:,:,:]=np.broadcast_to(I0[-(R+1),:,:],(R,*(I0.shape[1:])))
    I0[:,:R,:]=np.broadcast_to(I0[:,R,:][:,None,:],(I0.shape[0],R,I0.shape[2]))
    I0[:,-R:,:]=np.broadcast_to(I0[:,-(R+1),:][:,None,:],(I0.shape[0],R,I0.shape[2]))
    
    I1=I0[R:-R,2*R:,:]
    I3=I0[:-2*R,R:-R]
    I5=I0[R:-R,:-2*R,:]
    I7=I0[2*R:,R:-R,:]
    if N==8:
        I2=I0[:-2*R,2*R:,:]
        I4=I0[:-2*R,:-2*R]
        I6=I0[2*R:,:-2*R,:]
        I8=I0[2*R:,2*R:,:]
    L=(1-tau)*imgs
    U=(1+tau)*imgs
    
    if N==4:
        J=(I1<L)+(I1>U)*2+((I3<L)+(I3>U)*2)*3+((I5<L)+(I5>U)*2)*9+((I7<L)+(I7>U)*2)*27
    elif N==8:
        J=(I1<L)+(I1>U)*2+((I2<L)+(I2>U)*2)*3+((I3<L)+(I3>U)*2)*3**2+((I4<L)+(I4>U)*2)*3**3+\
            ((I5<L)+(I5>U)*2)*3**4+((I6<L)+(I6>U)*2)*3**5+((I7<L)+(I7>U)*2)*3**6+\
            ((I8<L)+(I8>U)*2)*3**7
    return J

if __name__=='__main__':
    pass
    