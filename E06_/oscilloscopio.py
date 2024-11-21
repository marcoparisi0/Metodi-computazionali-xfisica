import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse


data=pd.read_csv('oscilloscope.csv')
dati=data[['time','signal1','signal2']]
t=np.array(dati['time'])
c1=np.array(dati['signal1'])
c2=np.array(dati['signal2'])



def dv(xx, yy, nh):
    dd = yy[nh:] - yy[:-nh]
    hh = xx[nh:] - xx[:-nh]
    
    for ih in range(int(nh/2)):
        dd = np.append(yy[nh-ih-1]-yy[0], dd)
        dd = np.append(dd, yy[-1]-yy[-(nh-ih)])
    
        hh = np.append(xx[nh-ih-1]-xx[0], hh)
        hh = np.append(hh, xx[-1]-xx[-(nh-ih)])
    return dd/hh

nh=int(input('inserire la precisione con cui calcolare la derivata (intervallo da prendere per la dfferenza centrale per aggirare il rumore):'))




fig, axs = plt.subplots(2,2,figsize=(12,6))
axs[0,0].plot(t,c1,color= 'blue')
axs[0,0].set_xlabel('tempo')
axs[0,0].set_ylabel('voltaggio')
axs[0,0].set_title('segnale oscilloscopio CH1')

axs[0,1].plot(t,c2,color='green')
axs[0,1].set_xlabel('tempo')
axs[0,1].set_ylabel('voltaggio')
axs[0,1].set_title('segnale oscilloscopio CH2')

axs[1,0].plot(t,dv(t,c1,nh),color= 'red')
axs[1,0].set_xlabel('tempo')
axs[1,0].set_ylabel('voltaggio')
axs[1,0].set_title('derivata segnale oscilloscopio CH1')

axs[1,1].plot(t,dv(t,c2,nh),color='purple')
axs[1,1].set_xlabel('tempo')
axs[1,1].set_ylabel('voltaggio')
axs[1,1].set_title('derivata segnale oscilloscopio CH2')

plt.show()   
