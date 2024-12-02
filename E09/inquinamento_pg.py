
import numpy as np
import pandas as pd
from scipy import constants, fft, optimize
import matplotlib.pyplot as plt

data=pd.read_csv('copernicus_PG_selected.csv')

anno=data['date']
co=data['mean_co_ug/m3']
nh3=data['mean_nh3_ug/m3']
no2=data['mean_no2_ug/m3']
o3=data['mean_o3_ug/m3']
pm10=data['mean_pm10_ug/m3']
pm2p5=data['mean_pm2p5_ug/m3']
so2=data['mean_so2_ug/m3']

inquinanti=[co,nh3,no2,o3,pm10,pm2p5,so2]
colori=['goldenrod','limegreen','darkred','magenta','rebeccapurple','cyan','gray']
inq=['co','nh3','no2','o3','pm10','pm2p5','so2']

for i in range(len(inquinanti)):
    plt.plot(anno,inquinanti[i], color=colori[i],label=inq[i])

plt.xlabel('anno')
plt.ylabel('ug/m3')
plt.legend(fontsize=7)
plt.show()



tr_co=fft.rfft(co.values)
f_co=fft.rfftfreq(tr_co.size,d=anno[1]-anno[0])

mask= np.absolute(tr_co)**2 < 10**7
tr_co_filtrati=tr_co.copy()
tr_co_filtrati[mask]=0

filtered_co = fft.irfft(tr_co_filtrati, n=len(anno))



fig, axs = plt.subplots(2,2,figsize=(13,7))

axs[0,0].plot(f_co[:int(tr_co.size//2)],np.absolute(tr_co[:int(tr_co.size//2)])**2,color='indianred')
axs[0,0].set_xlabel('Frequenza')
axs[0,0].set_ylabel('|Ck|^2')
axs[0,0].set_xscale('log')
axs[0,0].set_yscale('log')
axs[0,0].set_title('spettro di potenza, f')

axs[0,1].plot(1/(f_co[1:int(tr_co.size//2)]),np.absolute(tr_co[1:int(tr_co.size//2)])**2,color='seagreen')
axs[0,1].set_xlabel('Periodo [anni]')
axs[0,1].set_ylabel('|Ck|^2')
axs[0,1].set_xscale('log')
axs[0,1].set_yscale('log')
axs[0,1].set_title('spettro di potenza, T')

axs[1,0].plot(anno,co, color ='darkorchid')
axs[1,0].set_title('grafico co originale')
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')
axs[1,0].set_xlabel('anni')
axs[1,0].set_ylabel('ug/m3')
axs[1,1].plot(anno,filtered_co, color='goldenrod')
axs[1,1].set_title('grafico co filtrato')
axs[1,1].set_xscale('log')
axs[1,1].set_yscale('log')
axs[1,1].set_xlabel('anni')
axs[1,1].set_ylabel('ug/m3')

plt.show()




