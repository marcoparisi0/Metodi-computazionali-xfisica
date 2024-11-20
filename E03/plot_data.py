import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dati=pd.read_csv('days.csv')

t=dati['temperatura']
g=dati['giorno']
t_err=dati['errore_T']
p=dati['mm_pioggia']
p_err=dati['errore_mm']

plt.plot(g,t,color='royalblue' )
plt.xlabel('giorno')
plt.ylabel('temperatura')
plt.show()

plt.errorbar(g,p,yerr=p_err, fmt= 'o', color = 'green')
plt.xlabel('giorno')
plt.ylabel('mm pioggia')
plt.show()

fig,ax = plt.subplots(1,2, figsize=(12,6))
ax[0].errorbar(g,t,yerr=t_err , color='royalblue', fmt= '*' )
ax[0].set_xlabel('giorno')
ax[0].set_ylabel('temperatura')
ax[1].errorbar(g,p,yerr=p_err, fmt= 'o', color = 'green')
ax[1].set_xlabel('giorno')
ax[1].set_ylabel('pioggia')
plt.show()

plt.scatter(t,p,color= 'red', alpha=0.5)
plt.xlabel('T')
plt.ylabel('mm pioggia')
plt.show()


