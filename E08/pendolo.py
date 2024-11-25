import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from  scipy import integrate
from scipy.constants import g

def drdt_pendolo(r, t,l ):
    """
    drdt_molla(r, t) derivate per equazine differenziale del moto oscillante attenuato di una molla 
    r : vettore con variabili r( teta , dteta/dt (ovvero w) )
    t : variabile tempo
   restituirò due variabili 
    """

    dtdt = r[1]
    dwdt = -(g/l)*np.sin(r[0])
    return (dtdt, dwdt)
time=np.linspace(0,20,100)



ci=(np.pi/4,0)
pol=integrate.odeint(drdt_pendolo,ci,time,args=(0.5,))

ci2=(np.pi/4,0)
pol2=integrate.odeint(drdt_pendolo,ci2,time,args=(1,))

ci3=(np.pi/6,0)
pol3=integrate.odeint(drdt_pendolo,ci3,time,args=(0.5,))


fig, axs = plt.subplots(1,3,figsize=(12,6))
axs[0].plot(time,pol[:,0],color= 'green',label= 'theta0= pi/4, l=0.5 m')
axs[0].set_xlabel('tempo')
axs[0].set_ylabel('angolo')
axs[0].set_title('pendolo1')
axs[0].legend(loc='lower right',fontsize=9)
axs[1].plot(time,pol2[:,0],color= 'blue',label= 'theta0= pi/4, l=1 m')
axs[1].set_xlabel('tempo')
axs[1].set_ylabel('angolo')
axs[1].set_title('pendolo2')
axs[1].legend(loc='lower right',fontsize=9)
axs[2].plot(time,pol3[:,0],color= 'gold',label= 'theta0= pi/6, l=0.5 m')
axs[2].set_xlabel('tempo')
axs[2].set_ylabel('angolo')
axs[2].set_title('pendolo3')
axs[2].legend(loc='lower right',fontsize=9)
plt.show()
