import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

link= "https://raw.githubusercontent.com/s-germani/metodi-computazionali-fisica-2024/refs/heads/main/dati/moduli_scientifici/kplr010666592-2011240104155_slc.csv" #metti le virgolette
data=pd.read_csv(link)

dati= data[['TIME', 'PDCSAP_FLUX','PDCSAP_FLUX_ERR']]
print(dati)
plt.plot(dati['TIME'], dati[ 'PDCSAP_FLUX'])
plt.xlabel('s')
plt.ylabel('flusso')
plt.show()
plt.plot(dati['TIME'], dati[ 'PDCSAP_FLUX'], 'o')
plt.show()
plt.errorbar(dati['TIME'], dati[ 'PDCSAP_FLUX'],yerr=dati['PDCSAP_FLUX_ERR'])
plt.show()
#per salavre il grafico su un file png o pdf devo usare plt.savefig('nomefile.png o pd')
#plt.savefig('datikepler.pdf')
minimo=dati['PDCSAP_FLUX'].min()#concettualmente non è il minimo vero pk son periodici , è una fluttuazione che mi identifica UN minimo di tanti, non quello vero, ma vabe, lui diceva di prendere semplicemente un valore osservando il grafico
mindata=dati.loc[(dati['PDCSAP_FLUX']>minimo)&(dati['PDCSAP_FLUX']<minimo+200)]
plt.plot(mindata['TIME'], mindata[ 'PDCSAP_FLUX'])
plt.show()
#così è come posiziono il grafico diversamente ,tipo riquadro
#subplots
fig,ax = plt.subplots( figsize=(12,6) )  # Plot diviso in 1 riga e due colonne (1,2)

'''ax[0].plot(dati['TIME'], dati[ 'PDCSAP_FLUX'], color='limegreen')
ax[1].plot(mindata['TIME'], mindata['PDCSAP_FLUX'],  color='red')

ax[0].set_title('Grafico normale', fontsize=15, color='limegreen')
ax[1].set_title('Grafico minimo', fontsize=5, color='red')

ax[0].set_xlabel('t')
ax[0].set_ylabel('flusso')

ax[1].set_xlabel('t')
ax[1].set_ylabel('flusso')

ax[0].tick_params(axis='x', labelsize=14)
ax[0].tick_params(axis='y', labelsize=14)

plt.show()'''
ins_ax = ax.inset_axes([0.7,0.1,0.25,0.25])
ins_ax.errorbar(mindata['TIME'], mindata[ 'PDCSAP_FLUX'],yerr=mindata['PDCSAP_FLUX_ERR'],fmt='.-', color='blue')
ax.set_xlabel('tempo')
ax.set_ylabel('flusso')
plt.errorbar(dati['TIME'], dati[ 'PDCSAP_FLUX'],yerr=dati['PDCSAP_FLUX_ERR'], color= 'purple')
plt.xlabel('tempo')
plt.ylabel('flusso')
plt.show()

