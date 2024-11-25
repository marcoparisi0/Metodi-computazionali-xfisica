import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from  scipy import integrate
import math



def v(t) :
    if np.isscalar(t):
        tt=int(t)
        if (tt%2 != 0):
            return -1
        if(tt%2 == 0):
            return 1
    else:
        v_in = np.ones(len(t)) 
        odd_mask = time.astype(int) %2 != 0
        v_in[odd_mask] = -1
        return v_in



def fode(v_out,t,RC):
    return (v(t)-v_out)/RC


init=0
time = np.linspace(0, 10, 100)

#con RC=4 
vv_4=integrate.odeint(fode,init,t=time,args=(4,))
#con RC=1
vv_1=integrate.odeint(fode,init,t=time,args=(1,))
#con RC=0.25 
vv=integrate.odeint(fode,init,t=time,args=(0.25,))

"""
odeint si aspetta come primo punto la funzione, i valori iniziali delle variabili(tutte), l'array dei tempi, poi gli altri parametri(che devono stare in una tupla
"""

fig, axs = plt.subplots(1,3,figsize=(12,6))
axs[0].plot(time,vv_4,color= 'green')
axs[0].set_xlabel('t')
axs[0].set_ylabel('V out')
axs[0].set_title('filtro passa basso  RC=4')
axs[1].plot(time,vv_1,color= 'blue')
axs[1].set_xlabel('t')
axs[1].set_ylabel('V out')
axs[1].set_title('filtro passa basso RC=1')
axs[2].plot(time,vv,color= 'gold')
axs[2].set_xlabel('t')
axs[2].set_ylabel('V out')
axs[2].set_title('filtro passa basso RC=0.25')
plt.show()



#salvataggio dati su CSV
dati = {
    "V con RC=4": vv_4[:,0],
    "V con RC=1": vv_1[:,0],
    "V con RC=1": vv[:,0],
    "tempi" : time,
    "V ingresso" : v(time)
}
#potevo usare .flatten che li appiattisce
df = pd.DataFrame(dati)
df.to_csv("output.csv", index=False)
