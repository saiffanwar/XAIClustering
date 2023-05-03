import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as lin

#generate some random data
N=100
x=np.random.rand(N,2)
x[:,1]=1

w=np.random.rand(2,2)
y=np.zeros(N)

n=int(np.random.rand()*N)
y[:n]=np.dot(x[:n,:],w[0,:])+np.random.normal(size=n)*.01
y[n:]=np.dot(x[n:,:],w[1,:])+np.random.normal(size=N-n)*.01


rx=np.ones( (100,2) )
r=np.arange(0,1,.01)
rx[:,0]=r

#plot the random dataset
plt.plot(x[:,0],y,'.b')
plt.plot(r,np.dot(rx,w[0,:]),':k',linewidth=2)
plt.plot(r,np.dot(rx,w[1,:]),':k',linewidth=2)

# regularization parameter for the regression weights
lam=.01

def em():
    # mixture weights
    rpi=np.zeros( (2) )+.5

    # expected mixture weights for each data point
    pi=np.zeros( (len(x),2) )+.5

    #the regression weights
    w1=np.random.rand(2)
    w2=np.random.rand(2)

    #precision term for the probability of the data under the regression function 
    eta=100

    for i in range(100):
        if 0:
            plt.plot(r,np.dot(rx,w1),'-r',alpha=.5)
            plt.plot(r,np.dot(rx,w2),'-g',alpha=.5)

        #compute lhood for each data point
        err1=y-np.dot(x,w1)
        err2=y-np.dot(x,w2)
        prbs=np.zeros( (len(y),2) )
        prbs[:,0]=-.5*eta*err1**2
        prbs[:,1]=-.5*eta*err2**2

        #compute expected mixture weights
        pi=np.tile(rpi,(len(x),1))*np.exp(prbs)
        pi/=np.tile(np.sum(pi,1),(2,1)).T

        #max with respect to the mixture probabilities
        rpi=np.sum(pi,0)
        rpi/=np.sum(rpi)

        #max with respect to the regression weights
        pi1x=np.tile(pi[:,0],(2,1)).T*x
        xp1=np.dot(pi1x.T,x)+np.eye(2)*lam/eta
        yp1=np.dot(pi1x.T,y)
        w1=lin.solve(xp1,yp1)

        pi2x=np.tile(pi[:,1],(2,1)).T*x
        xp2=np.dot(pi2x.T,x)+np.eye(2)*lam/eta
        yp2=np.dot(pi[:,1]*y,x)
        w2=lin.solve(xp2,yp2)

        #max wrt the precision term
        eta=np.sum(pi)/np.sum(-prbs/eta*pi)

        #objective function - unstable as the pi's become concentrated on a single component
        obj=np.sum(prbs*pi)-np.sum(pi[pi>1e-50]*np.log(pi[pi>1e-50]))+np.sum(pi*np.log(np.tile(rpi,(len(x),1))))+np.log(eta)*np.sum(pi)
        print(obj,eta,rpi,w1,w2)

        try:
            if np.isnan(obj): break
            if np.abs(obj-oldobj)<1e-2: break
        except:
            pass

        oldobj=obj

    return w1,w2


#run the em algorithm and plot the solution
rw1,rw2=em()
plt.plot(r,np.dot(rx,rw1),'-r')
plt.plot(r,np.dot(rx,rw2),'-g')

plt.savefig('./emExample.png')