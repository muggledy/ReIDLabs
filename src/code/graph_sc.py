'''
References:
[1] M. Zheng et al., "Graph Regularized Sparse Coding for Image Representation," in IEEE 
    Transactions on Image Processing, vol. 20, no. 5, pp. 1327-1336, May 2011, doi: 
    10.1109/TIP.2010.2090535.
muggledy 2020/5/20
'''

import numpy as np
from numpy.linalg import inv,pinv,matrix_rank
from itertools import count
from .cprint import cprint_out,cprint_err

def learn_dict(X,Y,c=1,max_iter=100):
    '''update D: min ||X-DY||_F^2 s.t. ||D(:,i)||_2^2<=c'''
    k=Y.shape[0]
    dual_lambds=np.abs(np.random.randn(k,1)) #any arbitrary initialization should be ok
    ### update dual_lambds
    XY_T=X.dot(Y.T)
    YY_T=Y.dot(Y.T)
    for i in count():
        if i>=max_iter:
            cprint_err('Newton Max Iter(%d)!'%max_iter)
            break
        YY_T_inv=pinv(YY_T+np.diag(dual_lambds.reshape(-1)))
        gradient=(np.sum((XY_T.dot(YY_T_inv))**2,axis=0)-c)[:,None]
        t=XY_T.dot(YY_T_inv)
        hessian=-2*(t.T.dot(t)*YY_T_inv)
        old_dual_lambds=dual_lambds
        #dual_lambds=dual_lambds-inv(hessian+0.001*np.eye(hessian.shape[0])).dot(gradient)
        dual_lambds=dual_lambds-pinv(hessian).dot(gradient) #
        #eps=0.0001
        #if np.sum((dual_lambds-old_dual_lambds)**2)<eps:
        if np.allclose(dual_lambds,old_dual_lambds):
            cprint_out('Newton Convergence(%d)!'%(i+1))
            break
    ### by Newton's method
    return XY_T.dot(YY_T_inv)

########################################################################################

def fobj_featuresign(s,B,x,BtB,Btx,P,L,i,alpha,gamma):
    f=0.5*np.sum((x-B.dot(s))**2)+alpha*L[i,i]*s.T.dot(s)+2*alpha*s.T.dot(P)
    f+=(gamma*np.sum(np.abs(s)))
    g=BtB.dot(s)+alpha*L[i,i]*s+alpha*P-Btx
    g+=(gamma*np.sign(s))
    return f,g

def compute_FS_step(s,B,x,BtB,Btx,P,L,idx,theta,act,act_indx1,alpha,gamma):
    #print(s.shape,B.shape,x.shape,BtB.shape,Btx.shape,P.shape,L.shape,theta.shape,act.shape,act_indx1.shape)
    s=s.copy()
    theta=theta.copy()
    act=act.copy()
    act_indx1=act_indx1.copy()
    s2=s[act_indx1,0]
    BtB2=BtB[np.ix_(act_indx1.flatten(),act_indx1.flatten())]
    #print(BtB.shape,act_indx1.shape,BtB2.shape)
    theta2=theta[act_indx1,0]
    a=L[idx,idx]*alpha*np.ones(act_indx1.shape)
    #print((BtB2+np.diag(a)).shape)
    s_new=inv(BtB2+np.diag(a)).dot(Btx[act_indx1,0]-gamma*theta2-alpha*P[act_indx1,0])
    optimality1=False
    if (np.sign(s_new)==np.sign(s2)).all():
        optimality1=True
        s[act_indx1,0]=s_new
        fobj=0
        lsearch=1
        return s,theta,act,act_indx1,optimality1,lsearch,fobj
    progress=-s2/(s_new-s2)
    lsearch=0
    a=0.5*np.sum((B[:,act_indx1.flatten()].dot(s_new-s2))**2)+ \
        0.5*alpha*L[idx,idx]*(s_new-s2).T.dot(s_new-s2)
    b=(s_new-s2).T.dot(alpha*P[act_indx1,0]-Btx[act_indx1,0])+(s2.T.dot(BtB2)+ \
        alpha*L[idx,idx]*s2.T).dot(s_new-s2)
    fobj_lsearch=gamma*np.sum(np.abs(s2))
    #print(progress.shape)
    t_lsearch=np.squeeze(np.hstack((progress.T,np.array([[1]]))))
    ix_lsearch=np.argsort(t_lsearch)
    #print(t_lsearch,ix_lsearch)
    sort_lsearch=t_lsearch[ix_lsearch]
    remove_idx=[]
    for i in range(len(sort_lsearch)):
        t=sort_lsearch[i]
        if t<=0 or t>1:
            continue
        s_temp=s2+(s_new-s2)*t
        fobj_temp=a*t**2+b*t+gamma*np.sum(np.abs(s_temp))
        if fobj_temp<fobj_lsearch:
            fobj_lsearch=fobj_temp
            lsearch=t
            if t<1:
                remove_idx=np.hstack((remove_idx,ix_lsearch[i]))
        elif fobj_temp>fobj_lsearch:
            break
        else:
            if np.sum(s2==0)==0:
                lsearch=t
                fobj_lsearch=fobj_temp
                if t<1:
                    remove_idx=np.hstack((remove_idx,ix_lsearch[i]))
    if lsearch>0:
        s_new=s2+(s_new-s2)*lsearch
        #print(s.shape,act_indx1.shape,s_new.shape)
        s[act_indx1.flatten()]=s_new
        theta[act_indx1.flatten()]=np.sign(s_new)
    eps=2.2204e-16
    if lsearch<1 and lsearch>0:
        remove_idx=np.where(np.abs(s[act_indx1])<eps)[0][:,None]
        s[act_indx1[remove_idx]]=0
        theta[act_indx1[remove_idx]]=0
        act[act_indx1[remove_idx]]=0
        act_indx1=act_indx1[sorted(list(set(range(len(act_indx1)))-set(remove_idx)))]
    fobj_new=fobj_featuresign(s,B,x,BtB,Btx,P,L,idx,alpha,gamma)
    fobj=fobj_new
    return s,theta,act,act_indx1,optimality1,lsearch,fobj

def ls_featuresign_sub(B,S,x,BtB,Btx,L,i,alpha,gamma,sinit=None):
    N,M=B.shape
    rankB=matrix_rank(BtB)
    usesinit=False
    if sinit is None:
        s=np.zeros((M,1))
        act=np.zeros((M,1))
        allowZero=False
    else:
        s=sinit
        theta=np.sign(s)
        act=np.abs(theta)
        usesinit=True
        allowZero=True
    L_new=L[:,[i]]
    L_new[i]=0
    P=S.dot(L_new)
    fobj=fobj_featuresign(s,B,x,BtB,Btx,P,L,i,alpha,gamma)
    ITERMAX=1000
    optimality1=False
    for iter in range(ITERMAX):
        act_indx0=np.where(act==0)[0][:,None]
        grad=BtB.dot(s)+alpha*L[i,i]*s+alpha*P-Btx
        theta=np.sign(s)
        optimality0=False
        t_mx=np.abs(grad[act_indx0])
        indx=np.argmax(t_mx) if len(t_mx)>0 else None
        mx=t_mx[indx] if indx!=None else None
        if mx!=None and mx>=gamma and (iter>0 or not usesinit):
            act[act_indx0[indx]]=1
            theta[act_indx0[indx]]=-np.sign(grad[act_indx0[indx]])
            usesinit=False
        else:
            optimality0=True
            if optimality1:
                break
        act_indx1=np.where(act==1)[0][:,None]
        if len(act_indx1)>rankB:
            print('WARNNING: sparsity penalty is too small: too many coefficients are activated')
            return s,fobj
        if len(act_indx1)==0:
            if allowZero:
                allowZero=False
                continue
            return s,fobj
        k=0
        while 1:
            k+=1
            if k>ITERMAX:
                print('WARNNING: Maximum number of iteration reached. The solution may not be optimal')
                return s,fobj
            if len(act_indx1)==0:
                if allowZero:
                    allowZero=False
                    break
                return s,fobj
            s,theta,act,act_indx1,optimality1,lsearch,fobj=compute_FS_step(s,B,x,BtB,Btx,P,L,i,theta,act,act_indx1,alpha,gamma)
            if optimality1:
                break
            if lsearch>0:
                continue
    if iter>=ITERMAX:
        print('WARNNING: Maximum number of iteration reached. The solution may not be optimal')
    fobj=fobj_featuresign(s,B,x,BtB,Btx,P,L,i,alpha,gamma)
    return s,fobj

def learn_coding(B,X,alpha,gamma,L,Sinit=None):
    '''update S: min 0.5*(||Xi-B*Si||^2 + alpha*Lii*SiᵀSi + 2*alpha*siᵀ(∑Lij*sj) 
       + gamma*||si||_1 (feature sign search)
       Sinit is the initial coefficient matrix
       Note: this func translated from author(Miao Zheng)'s source code(matlab) 
       directly, but it's quality is hard to say(not good!)'''
    use_Sinit=False if Sinit is None else True
    Sout=np.zeros((B.shape[1],X.shape[1]))
    BtB=B.T.dot(B)
    BtX=B.T.dot(X)
    rankB=matrix_rank(BtB)
    for i in range(X.shape[1]):
        if use_Sinit:
            idx1=np.where(Sinit[:,i]!=0)[0]
            maxn=np.minimun(len(idx1),rankB)
            sinit=np.zeros(Sinit[:,[i]].shape)
            sinit[idx1[:maxn]]=Sinit[idx1[:maxn],[i]]
            a=np.sum(np.sum(Sinit,axis=1)==0)
            S=np.hstack((Sout[:,:i],Sinit[:,i:]))
            Sout[:,[i]],fobj=ls_featuresign_sub(B,S,X[:,[i]],BtB,BtX[:,[i]],L,i, \
                alpha,gamma,sinit)
        else:
            Sout[:,[i]],fobj=ls_featuresign_sub(B,Sout,X[:,[i]],BtB,BtX[:,[i]],L,i, \
                alpha,gamma)

def getObjective(B,S,X,alpha,L,noise_var,beta,sigma):
    lambd=1/noise_var
    fresidue=0.5*lambd*np.sum((B.dot(S)-X)**2)
    flaplacian=0.5*alpha*np.trace(S.dot(L).dot(S.T))
    fsparsity=beta*np.sum(np.abs(S/sigma))
    fobj=fresidue+fsparsity+flaplacian
    return fobj,fresidue,fsparsity,flaplacian

def graph_sc(X,W,num_bases,alpha,beta,num_iters,Binit=None,c=1):
    '''Graph regularized sparse coding algorithms
       minimize_B,S   0.5*||X - B*S||^2 + alpha*Tr(SLS') + beta*sum(abs(S(:)))
       subject to   ||B(:,j)||_2 <= c, forall j=1...size(S,1)
       Args:
       X: data matrix, each column is a sample vector
       W: affinity graph matrix
       num_bases: number of bases
       alpha: Laplician parameter
       beta: sparsity penalty parameter
       num_iters: number of iteration
       Binit: initial B matrix
       Notes:
       Also translated from author's source code(matlab)'''
    diff=1e-7
    mFea,nSmp=X.shape
    noise_var=1
    sigma=1
    if Binit is None:
        B=np.random.rand(mFea,num_bases)-0.5
        B=B-np.mean(B,axis=0)
        B=B.dot(np.diag(1/np.sqrt(np.sum(B*B,axis=0))))
    else:
        print('Using Binit...')
        B=Binit
    t=0
    stat={}
    stat['fobj_avg']=[]
    stat['fresidue_avg']=[]
    stat['fsparsity_avg']=[]
    stat['flaplacian_avg']=[]
    D=np.diag(np.sum(W,axis=1))
    L=D-W
    while t<num_iters:
        t+=1
        stat['fobj_total']=0
        stat['fresidue_total']=0
        stat['fsparsity_total']=0
        stat['flaplacian_total']=0
        if t==1:
            S=learn_coding(B,X,alpha,beta/sigma*noise_var,L)
        else:
            S=learn_coding(B,X,alpha,beta/sigma*noise_var,L,S)
        print(S.shape)
        print(S.dtype)
        print(S)
        #S[np.isnan(S)]=0
        fobj,fresidue,fsparsity,flaplacian= \
            getObjective(B,S,X,alpha,L,noise_var,beta,sigma)
        stat['fobj_total']=stat['fobj_total']+fobj
        stat['flaplacian_total']=stat['flaplacian_total']+flaplacian
        stat['fresidue_total']=stat['fresidue_total']+fresidue
        stat['fsparsity_total']=stat['fsparsity_total']+fsparsity
        B=learn_dict(X,S,c)
        stat['fobj_avg'].append(stat['fobj_total']/nSmp)
        stat['fresidue_avg'].append(stat['fresidue_total']/nSmp)
        stat['fsparsity_avg'].append(stat['fsparsity_total']/nSmp)
        stat['flaplacian_avg'].append(stat['flaplacian_total']/nSmp)
        if t>199:
            if stat['fobj_avg'][t-1]-stat['fobj_avg'][t]<diff:
                return B,S,stat
        print('epoch= %d, fobj= %f, fresidue= %f, flaplacian= %f, fsparsity= %f'% \
            (stat['fobj_avg'][t],stat['fresidue_avg'][t],stat['flaplacian_avg'][t], \
            stat['fsparsity_avg'][t]))
        