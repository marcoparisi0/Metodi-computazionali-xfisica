import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_0= pd.read_csv( 'hit_times_M0.csv' )
data_1= pd.read_csv( 'hit_times_M1.csv' )
data_2= pd.read_csv( 'hit_times_M2.csv' )
data_3= pd.read_csv( 'hit_times_M3.csv' )

dati_0= data_0[['mod_id','det_id','hit_time']]
dati_1= data_1[['mod_id','det_id','hit_time']]
dati_2= data_2[['mod_id','det_id','hit_time']]
dati_3= data_3[['mod_id','det_id','hit_time']]


plt.hist(dati_0['hit_time'], bins=100, color='green', alpha=0.7 )
plt.xlabel('tempo di hit')
plt.ylabel('n.volte')
plt.title('istogramma dei tempi di hit [s]')
plt.show()

delta_t_0=np.diff(dati_0['hit_time']) # Calcola la differenza tra valori consecutivi nell'array

plt.hist(np.log10(delta_t_0), bins=50,range=(-0.5,8), color='purple', alpha=0.6)
"""
il logaritmo serve per visualizzare l'amdamento di valori grandissimi, ma in scala piccola
"""

plt.xlabel(' diff tempo di hit (log10)')
plt.ylabel('n.volte')
plt.title('istogramma delle differenze tra i  tempi di hit')
plt.show()
