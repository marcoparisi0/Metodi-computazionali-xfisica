import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse


data=pd.read_csv('oscilloscope.csv')
dati=data[['time','signal1','signal2']]
t=dati['time']
c1=dati['signal1']
c2=dati['signal2']

plt.plot(t,c1,color= 'blue')
plt.xlabel('tempo')
plt.ylabel('voltaggio')
plt.title('segnale oscilloscopio CH1')
plt.show()
plt.plot(t,c2,color='green')
plt.xlabel('tempo')
plt.ylabel('voltaggio')
plt.title('segnale oscilloscopio CH2')
plt.show()


def dv(xx, yy, nn):
    d=[]
    """for i in range(int(nn/2),len(t)-int(nn/2)):
        cucu=(yy[i+(nn/2)] - yy[i-(nn/2)])/(xx[i+(nn/2)] - xx[i-(nn/2)])"""
    for i in range(0,len(t)+1):
        cucu=(yy[i+(nn)] - yy[i])/(xx[i+(nn/2)] - xx[i-(nn/2)])
    
        d.append(cucu)
    return d

nh=int(input('inserire la precisione con cui calcolare la derivata (intervallo da prendere per la dfferenza centrale per aggirare il rumore):'))

plt.plot(t,dv(t,c1,nh),color= 'red')
plt.xlabel('tempo')
plt.ylabel('voltaggio')
plt.title('derivata segnale oscilloscopio CH1')
plt.show()
plt.plot(t,dv(t,c2,nh),color='purple')
plt.xlabel('tempo')
plt.ylabel('voltaggio')
plt.title('derivata segnale oscilloscopio CH2')
plt.show()   
