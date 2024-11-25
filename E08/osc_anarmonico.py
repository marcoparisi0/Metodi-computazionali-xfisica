import numpy as np
import matplotlib.pyplot as plt
from  scipy import integrate
from scipy.constants import g

def drdt(r,t,o):
    dxdt=r[1]
    dydt=-(o**2)*(r[0]**3)
    return(dxdt,dydt)

time=np.linspace(0,50,250)

ini=(5,0)
x=integrate.odeint(drdt,ini,time,args=(3,))

inii=(1,0)
xx=integrate.odeint(drdt,inii,time,args=(3,))

iniii=(1,0)
xxx=integrate.odeint(drdt,iniii,time,args=(0.25,))

iniiii=(1,0)
xxxx=integrate.odeint(drdt,iniiii,time,args=(1,))

oj=(2.5,0)
hu=integrate.odeint(drdt,oj,time,args=(1,))

fig, axs = plt.subplots(2,2,figsize=(12,6))
axs[0,0].plot(time,x[:,0],color= 'red')
axs[0,1].plot(time,xx[:,0],color='purple')
axs[1,1].plot(time,xxx[:,0],color= 'blue')
axs[1,0].plot(time,xxxx[:,0],color='gold')
plt.show()

plt.plot(time,x[:,0],color= 'orchid', label='x0=5,w=3')
plt.plot(time,xx[:,0],color='olive', label='x0=1,w=3')
plt.plot(time,xxx[:,0],color= 'cornflowerblue',label='x0=1, w=0.25')
plt.plot(time,xxxx[:,0],color='firebrick',label='x0=1,w=1')
plt.plot(time,hu[:,0],color='navy',label='x0=2.5,w=1')
plt.legend(fontsize=9)
plt.show()




