import numpy as np
import pandas as pd
from scipy import constants, fft, optimize
import matplotlib.pyplot as plt

data1=pd.read_csv('data_sample1.csv')
data2=pd.read_csv('data_sample2.csv')
data3=pd.read_csv('data_sample3.csv')
t1=np.array(data1['time'])
v1=np.array(data1['meas'])
t2=np.array(data2['time'])
v2=np.array(data2['meas'])
t3=np.array(data3['time'])
v3=np.array(data3['meas'])


tr1=fft.rfft(v1)     #le trasformagte di fourier vogliono dentro un array
tr2=fft.rfft(v2)
tr3=fft.rfft(v3)
f1=fft.rfftfreq(len(tr1),d=t1[1]-t1[0])
f2=fft.rfftfreq(len(tr2),d=t2[1]-t2[0])
f3=fft.rfftfreq(len(tr3),d=t3[1]-t3[0])

#il fit non deve includere il termine in 0 perchènon identifica l'andamento in funzione del tempo, ma solo il valor medio, in più deve contenre un fattore di normalizzazione

def rum(f,b,k):
    return k/(f**b)


p0=(1,1)


param1, param1_covariance = optimize.curve_fit(rum,f1[10:int(tr1.size)//2], np.absolute(tr1[10:int(tr1.size//2)])**2, p0)
param2, param2_covariance = optimize.curve_fit(rum,f2[10:int(tr2.size)//2], np.absolute(tr2[10:int(tr2.size)//2])**2, p0)
param3, param3_covariance = optimize.curve_fit(rum,f3[10:int(tr3.size)//2], np.absolute(tr3[10:int(tr3.size)//2])**2, p0)

print('b1= ',param1[0],'   b2=', param2[0],'  b3=', param3[0])





"""
grafico rumore
"""
fig, axs = plt.subplots(1,3,figsize=(12,6))
axs[0].plot(t1,v1,color='limegreen')
axs[1].plot(t2,v2,color='magenta')
axs[2].plot(t3,v3,color='blue')
plt.show()

"""
grafico scatter spettro di potenza con fit
nel plot del fit, escludo il primo punto altrimenti "divide per zero"
"""
fig, axs = plt.subplots(1,3,figsize=(12,6))
axs[0].scatter(f1[:int(tr1.size//2)],np.absolute(tr1[:int(tr1.size//2)])**2,color='lightgreen',alpha=0.7)
axs[0].plot(f1[1:int(tr1.size//2)],rum(f1[1:int(tr1.size//2)],param1[0],param1[1]), color='forestgreen')
axs[0].set_xlabel('Frequenza [Hz]')
axs[0].set_ylabel('|Ck|^2')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_title('White noise')

axs[1].scatter(f2[:int(tr2.size//2)],np.absolute(tr2[:int(tr2.size//2)])**2,color='magenta',alpha=0.7)
axs[1].plot(f2[1:int(tr2.size//2)],rum(f2[1:int(tr2.size//2)],param2[0],param2[1]), color='darkred')
axs[1].set_xlabel('Frequenza [Hz]')
axs[1].set_ylabel('|Ck|^2')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_title('Pink noise')

axs[2].scatter(f3[:int(tr3.size//2)],np.absolute(tr3[:int(tr3.size//2)])**2,color='cornflowerblue', alpha=0.7)
axs[2].plot(f3[1:int(tr3.size//2)],rum(f3[1:int(tr3.size//2)],param3[0],param3[1]), color='navy')
axs[2].set_xlabel('Frequenza [Hz]')
axs[2].set_ylabel('|Ck|^2')
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_title('Brownian noise')

#prendo metà dei coefficienti perchè nel caso reale, sono identici, nel caso compless sono l'uno il complesso coniug dell altro, ma tanto faccio il modulo quadro

plt.show()




