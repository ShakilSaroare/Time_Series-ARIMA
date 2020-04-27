# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:14:37 2020

@author: Shakil
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt


def adf_test(data):
    k=int((len(data)-1)**(1/3))
    k=k+1
    x=my_embed(data,k)
    x1=x[:,1].reshape((-1,1))
    x=-np.diff(x)
    y=x[:,0]
    x=x[:,1:]
    x=np.hstack((x1,x))
    trend=np.reshape(np.arange(0,x.shape[0],1),(-1,1)) #Adding trend into the model
    x=np.hstack((trend,x))                                          # Addign trend as an explanatory variable
    x=np.hstack((np.ones([x.shape[0],1]),x))
    beta=np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(x),x)),np.matmul(np.matrix.transpose(x),y))
    res=y-np.matmul(x,beta)
    sd=np.dot(np.matrix.transpose(res),res)/(len(y)-len(beta))
    se=sd*np.linalg.inv(np.matmul(np.matrix.transpose(x),x))
    t=beta[2]/(se[2,2]**0.5)
    return t

def my_acf (data):
    ACF=np.zeros(30)
    ACF[0]=stats.pearsonr(data,data)[0]
    for i in range(1,len(ACF)):
        lag=data[i:]
        new_dat=data[:len(lag)]
        ACF[i]=stats.pearsonr(new_dat,lag)[0]

    # var_acf=1/(len(np.diff(ts1)));              ##this contant variance is used in R 

    var_acf = np.ones(30) / len(data)
    var_acf[0] = 0
    var_acf[1] = 1. / len(data)
    var_acf[2:] *= 1 + 2 * np.cumsum(ACF[1:-1]**2) # In python, Bartlett's formula is used

    plt.bar(range(len(ACF)), ACF, width=0, ec="k", capstyle="round", linewidth=1.5);
    plt.scatter(range(len(ACF)), ACF, color="dodgerblue", s=20)
    plt.fill_between(range(len(ACF)), (1.96*var_acf**0.5), -(1.96*var_acf**0.5),color="dodgerblue", alpha=0.2)
    plt.plot(range(len(ACF)),np.zeros(30),linestyle='solid',color="dodgerblue",linewidth=1);
    return plt.show()


def my_pacf (data):
    PACF=np.zeros(25)
    PACF[0]=1
    for k in range(1,len(PACF)):
        y=np.zeros(len(data)-k)
        x=np.zeros([len(data)-k,k])
        for i in range(len(data)-k):
            y[i]=data[i]
            x[i,:]=data[i+1:i+k+1]
        theta=np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(x),x)),np.matmul(np.matrix.transpose(x),y))
        PACF[k]=theta[k-1]

    var_pacf = np.ones(25) / len(data)
    var_pacf[0] = 0
    var_pacf[1] = 1. / len(data)
    var_pacf[2:] *= 1 + 2 * np.cumsum(PACF[1:-1]**2) # In python, Bartlett's formula is used

    plt.bar(range(len(PACF)), PACF, width=0, ec="k", capstyle="round", linewidth=1.5);
    plt.scatter(range(len(PACF)), PACF, color="dodgerblue", s=20)
    plt.fill_between(range(len(PACF)), (1.96*var_pacf**0.5), -(1.96*var_pacf**0.5),color="dodgerblue", alpha=0.2)
    plt.plot(range(len(PACF)),np.zeros(25),linestyle='solid',color="dodgerblue",linewidth=1);
    return plt.show()


def my_embed(x,m):
    x=np.array(x)
    y=np.zeros((len(x)+m,m+1))
    for i in range(0,m+1):
        y[m-i:len(x)+m-i,m-i]=x
    return y[m:-m]


class tseries:
    
    
    def __init__(self,data,order):
        
        self.data=data
        self.order=order
        
        
    def start_par(self):
        
        p=self.order[0]
        d=self.order[1]
        q=self.order[2]
        
        data=self.data
    
        for i in (range(d)):
            data=np.diff(data)

        if q>0:
            if p>0:
                maxlag=int(round(12 * (len(data) / 100.) ** (1 / 4.))) 

                x=my_embed(data,maxlag)
                y=x[:,0]
                x=x[:,1:]

                x=np.hstack((np.ones([x.shape[0],1]),x))

                theta=np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(x),x)),np.matmul(np.matrix.transpose(x),y))

                res=y-np.matmul(x,theta)

                res=np.hstack((np.zeros(q),res))

                z=my_embed(res,q)[:,1:]

                x=my_embed(data,p)[maxlag-p:,1:]

                x=np.hstack((np.ones([x.shape[0],1]),x))

                x=np.hstack((x,z))

                params=np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(x),x)),np.matmul(np.matrix.transpose(x),y))

            else:
                x=my_embed(data,q)
                y=x[:,0]
                x=x[:,1:]

                x=np.hstack((np.ones([x.shape[0],1]),x))

                params=np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(x),x)),np.matmul(np.matrix.transpose(x),y))

        else:
            x=my_embed(data,p)
            y=x[:,0]
            x=x[:,1:]

            x=np.hstack((np.ones([x.shape[0],1]),x))

            params=np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(x),x)),np.matmul(np.matrix.transpose(x),y))



        return params
    
    
    
    def ARIMA_css(self,par):
        
        p=self.order[0]
        d=self.order[1]
        q=self.order[2]
        
        data=self.data

        for i in (range(d)):
            data=np.diff(data)

        phi=np.array(par[1:p+1])
        theta=np.array(par[p+1:])
        res=np.zeros(len(data))

        if p>0:
            x=my_embed(data,p)
            y=x[:,0]-par[0]
            x=x[:,1:]-par[0]

            if q>0:
                for i in range(max(p,q),len(data)):
                    res[i]=y[i-max(p,q)]-np.dot(x[i-max(p,q),:],phi)-np.dot(res[i-q:i][::-1],theta)


                z=my_embed(res,q)[:,1:]
                if p>q:
                    z=z[(p-q):,:]
                else:
                    if q>p:
                        for i in range(q-p):
                            z=np.vstack((np.zeros(q),z))
                obj_fun=np.dot(y-np.dot(x,phi)-np.dot(z,theta),y-np.dot(x,phi)-np.dot(z,theta))

            else:
                obj_fun=np.dot(y-np.dot(x,phi),y-np.dot(x,phi))
        else:
            if q>0:
                y=data[q:]-par[0]
                for i in range(max(p,q),len(data)):
                    res[i]=y[i-max(p,q)]-np.dot(res[i-q:i][::-1],theta)
                z=my_embed(res,q)[:,1:]
                obj_fun=np.dot(y-np.dot(z,theta),y-np.dot(z,theta))


        return obj_fun
    
    
    def residual(self,par):
        
        data=self.data
        p=self.order[0]
        d=self.order[1]
        q=self.order[2]
        
        for i in (range(d)):
            data=np.diff(data)

        phi=np.array(par[1:p+1])
        theta=np.array(par[p+1:])
        resid=np.zeros(len(data))

        #z=np.zeros([len(data)-max(p,q),q])
        y=np.zeros(len(data)-max(p,q))
        x=np.zeros((len(data)-max(p,q),p))

        for i in range(max(p,q),len(data)):
            y[i-max(p,q)]=data[i]-par[0]
            for j in range(p):
                x[i-max(p,q),j]=data[i-j-1]-par[0]
            resid[i]=y[i-max(p,q)]-np.dot(x[i-max(p,q),:],phi)-np.dot(resid[i-q:i][::-1],theta)
                
        return resid

    
    def est_par(self):
        
        p=self.order[0]
       # q=self.order[2]
        
        mod=minimize(self.ARIMA_css,self.start_par())
        
        res=self.residual(mod.x)
        sig=(np.dot(res,res)/(len(res)-len(mod.x)))**0.5
        
        if mod.message =='Optimization terminated successfully.':
            print("Intercept:", np.around([mod.x[0]],10))
            print("AR Parameter(s):", np.around(mod.x[1:p+1],4))
            print("MA Parameter(s):", np.around(mod.x[p+1:],4))
            print("Standard Deviation:", np.around(sig,4))
        else:
            print("The css method didn't converge")