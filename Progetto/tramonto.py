import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from  scipy import integrate
import matplotlib.colors as mcolors
from scipy.constants import c,h,k, pi
import math


"""



il problema che avevo nell'esponente, è che, usando np.exp(arg)-1  ppure np.expm1(arg), veniva calcolato e^... , poi sottratto 1 e poi fatta la divisione. Questo causava overflow nel calcolo dell'esponente, essendo un numero moolto grosso.Dunque, essendo l'argomento dell'esponente >>1, ho considerato direttamente np.exp(-arg) che NON IMPLICA il calcolo di un exp con esponente gigante.  Spero vada bene



DA FARE:

creare un modulo con tutte le funzioni? e poi file diversi per le diverse stelle e per il fit??    Anche un qualcosa con argparse non sarebbe maleee, oppure una classe??
? ripetere cose specifiche fatte per il sole anche per le altre stelle?
"""

#DEFINIZIONE FUNZIONI E VALORI

T_sun=5.75*(10**3)   #K
T_sau=1.9*(10**3)    #K
T_vega=10*(10**3)    #K
T_rigel=25*(10**3)   #K

#parametri terrestri
R_T=6378000 # [m]   , Raggio terrestre
n_T=1.00029  #indice di rifrazione atmosfera terrestre
N_T=2.504*(10**25) # densità di molecole terrestre
S_z=8000 #m
S_o=np.sqrt((R_T+S_z)**2 -  R_T**2)


def S_teta(t):
     
    """
    Funzione che descrive lo spessore di massa di aria dipendentemente da un angolo generico
    Parametri
    -----------
        t: Angolo
    
    Restituisce
        np.sqrt((R_T*np.cos(t))**2 +  2*R_T*S_z + S_z**2) - R_T*np.cos(t)
    -----------
 
    """
    return np.sqrt((R_T*np.cos(t))**2 +  2*R_T*S_z + S_z**2) - R_T*np.cos(t)



def B(l,T):
     """
    Funzione che descrive la densità di energia irradiata da una stella
    Parametri
    -----------
        l: Lunghezza d'onda
        T: Temperatura della stella
    
    Restituisce
        una versione approssimata,che risolve l'overflow dovuto al calcolo di np.exp
     formula usata:    2*h*(c**2)*np.exp(-h*c/(l*k*T))/(l**5)
     formula vera:    2*h*(c**2)/(np.expm1(h*c/(l*k*T))*(l**5))
        
    -----------
 
    """
     return 2*h*(c**2)*np.exp(-h*c/(l*k*T))/(l**5)    



def D(l,T):
     """
    Funzione che descrive la densità di fotoni per lunghezza d'onda 
    Parametri
    -----------
        l: Lunghezza d'onda
        T: Temperatura della stella
    
    Restituisce
        una versione approssimata,che risolve l'overflow dovuto al calcolo di np.exp
     formula usata:   2*c*np.exp(-h*c/(l*k*T))/(l**4)
     formula vera:   2*c/((l**4)*np.expm1(h*c/(l*k*T)))
        
    -----------
 
    """
     return 2*c*np.exp(-h*c/(l*k*T))/(l**4)    



def beta(l):
     """
    Funzione che descrive la  probabilità di interazione dei fotoni con l'atmosfera terrestre
    Parametri
    -----------
        l: Lunghezza d'onda
    
    Restituisce
        
        8*pow(np.pi,3)*pow(n_T**2 - 1,2)/(3*N_T*l**4)
    -----------
 
    """
     return 8*pow(np.pi,3)*pow(n_T**2 - 1,2)/(3*N_T*l**4)


def D_scatter(l,s,T):
     """
    Funzione che descrive la  densità di fotoni per lunghezza d'onda modificata con la probabilità di interazione dei fotoni con l'atmosfera terrestre
    Parametri
    -----------
        l: Lunghezza d'onda
    
    Restituisce
        La distribuzione di fotoni solari considerando lo scattering di Rayleigh
        D(l,T)*np.exp(-beta(l)*s)
    -----------
 
    """
     return  D(l,T)*np.exp(-beta(l)*s)








la=np.random.uniform(low=10**(-8), high=5*10**(-6), size=10000)  # in  m   considero spettro da UV a infrarossi


plt.hist(la, bins=25, alpha=0.8, color='green', ec='darkgreen')
plt.title('Distribuzione uniforme lambda')
plt.show()

lmbd=np.sort(la)



#PLOT SPETTRO  SOLE
E=B(lmbd,T_sun)
plt.plot(lmbd,E, color='black')
plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')

# Filtrare la parte visibile dello spettro
mask = (lmbd >= 380e-9) & (lmbd <= 750e-9)
lmbd_vis = lmbd[mask]
E_vis = E[mask]

# Creazione della colormap basata sulle lunghezze d'onda 
cmap = plt.get_cmap("rainbow")  
norm = mcolors.Normalize(vmin=380e-9, vmax=750e-9)  # Normalizzazione solo per il visibile
colors = cmap(norm(lmbd_vis))
for i in range(len(lmbd_vis) - 1):
    plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [E_vis[i], E_vis[i + 1]], color=colors[i])
plt.show()





#DISTRIBUZIONE FOTONI SOLE ORIZZONTE
Tr=D_scatter(lmbd,S_o,T_sun)
Tr_vis = Tr[mask]

def D_scatter_min_o(l):
    return -D_scatter(l,S_o,T_sun)
ris_o=optimize.minimize(D_scatter_min_o,x0=8*(10**-7))
l_max_o=ris_o.x[0]
o_max=D_scatter(l_max_o,S_o,T_sun)


plt.plot(lmbd,Tr, color='black')
plt.plot(l_max_o,o_max,'o',color='darkslateblue',label=r'\lambda$  = {:.2e} m'.format(l_max_o))
plt.plot([l_max_o, l_max_o], [0, o_max], linestyle='dashed', color='darkslateblue', linewidth=1)
plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
plt.legend()

for i in range(len(lmbd_vis) - 1):
    plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Tr_vis[i], Tr_vis[i + 1]], color=colors[i])
plt.show()






#SOLE ZENITH
Z=D_scatter(lmbd,S_z,T_sun)
Z_vis = Z[mask]

def D_scatter_min_z(l):
    return -D_scatter(l,S_z,T_sun)
ris_z=optimize.minimize(D_scatter_min_z,x0=5*(10**-7))
l_max_z=ris_z.x[0]
z_max=D_scatter(l_max_z,S_z,T_sun)

plt.plot(lmbd,Z, color='black')
plt.plot(l_max_z,z_max,'o', color='darkslateblue',label=r'\lambda$  = {:.2e} m'.format(l_max_z))
plt.plot([l_max_z, l_max_z], [0, z_max], linestyle='dashed', color='darkslateblue', linewidth=1)
plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
plt.legend()
for i in range(len(lmbd_vis) - 1):
    plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Z_vis[i], Z_vis[i + 1]], color=colors[i])
plt.show()







#PLOT ALTRE STELLE
fig,axs = plt.subplots(2,4, figsize=(12,6))
axs[0,0].plot(lmbd,B(lmbd,T_sun),color='darkgoldenrod')
axs[0,1].plot(lmbd,B(lmbd,T_sau),color='navy')
axs[0,2].plot(lmbd,B(lmbd,T_vega),color='darkgreen')
axs[0,3].plot(lmbd,B(lmbd,T_rigel),color='firebrick')
axs[1,0].plot(lmbd,D(lmbd,T_sun), color='goldenrod')
axs[1,0].plot(lmbd,D_scatter(lmbd,S_z,T_sun),  color='gold')
axs[1,0].plot(lmbd,D_scatter(lmbd,S_o,T_sun),  color='tan')
axs[1,1].plot(lmbd,D(lmbd,T_sau), color='slateblue')
axs[1,1].plot(lmbd,D_scatter(lmbd,S_z,T_sau),  color='royalblue')
axs[1,1].plot(lmbd,D_scatter(lmbd,S_o,T_sau),  color='darkorchid')
#axs[1,1].set_yscale('log')    #slateblue e royalblue sono sovrapposti, quasi
axs[1,2].plot(lmbd,D(lmbd,T_vega), color='seagreen')
axs[1,2].plot(lmbd,D_scatter(lmbd,S_z,T_vega),  color='lime')
axs[1,2].plot(lmbd,D_scatter(lmbd,S_o,T_vega),  color='forestgreen')
axs[1,3].plot(lmbd,D(lmbd,T_rigel), color='brown')
axs[1,3].plot(lmbd,D_scatter(lmbd,S_z,T_rigel),  color='lightcoral')
axs[1,3].plot(lmbd,D_scatter(lmbd,S_o,T_rigel),  color='darkred')
plt.show()






"""
DA FARE:

individuare il picco di ognuno

"""






"""
per il flusso integrato integro D(l,T) tra i limiti scelti di lambda
"""



te=np.random.uniform(low=0,high=pi,size=1000)


plt.hist(te, bins=25, alpha=0.8, color='violet', ec='darkviolet')
plt.title(r'Distribuzione uniforme angoli [0,$\pi$]')
plt.show()


teta=np.sort(te)






f=[]
for t in teta:
    integranda=D_scatter(lmbd,S_teta(t),T_sun)
    flusso=integrate.simpson(integranda,dx=0.01)
    f.append(flusso)

w=[]
for t in teta:
    integranda=D_scatter(lmbd,S_teta(t),T_sau)
    flusso=integrate.simpson(integranda,dx=0.01)
    w.append(flusso)
v=[]
for t in teta:
    integranda=D_scatter(lmbd,S_teta(t),T_vega)
    flusso=integrate.simpson(integranda,dx=0.01)
    v.append(flusso)

r=[]
for t in teta:
    integranda=D_scatter(lmbd,S_teta(t),T_rigel)
    flusso=integrate.simpson(integranda,dx=0.01)
    r.append(flusso)





fig,axs = plt.subplots(2,2, figsize=(12,6))
axs[0,0].plot(S_teta(teta),f, color='teal')
axs[0,0].set_xscale('log')
axs[0,0].set_yscale('log')
axs[0,1].plot(S_teta(teta),w, color='pink')
axs[0,1].set_xscale('log')
axs[0,1].set_yscale('log')
axs[1,0].plot(S_teta(teta),v, color='violet')
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')
axs[1,1].plot(S_teta(teta),r, color='sienna')
axs[1,1].set_xscale('log')
axs[1,1].set_yscale('log')
plt.show()






"""
FINO A QUI OKK    (considerando buona l'approssimazione dell'esponenziale)
DA FARE
fare meglio i grafici e spiegare andamenti
"""




























"""
STELLA X

faccio il fit con la formula D_scatter, utilizzando photons, invece di modificare photons e fare il fit con D  (credo vada bene lo stesso). uso formula B, VA BENEE?---> SI VA BENE COGLIONE LEGGI LA CONSEGNA E LE COSE SOTTOLINEATE !!!

cose da fare:
chi quadro--(aggiustarlo, errori?)
spiegare (in caso nel ppt) come sono scelti  sti parametri, inizialmente p0=[6000, 1e-10], SOPRATTUTTO PERCHÈ CON ALCUNI VALORI MI VA IN OVERFLOW, CON ALTRI NO

FINO A QUI OKK
"""




dati=pd.read_csv('observed_starX.csv')
#print(dati)
l_nm=dati['lambda (nm)']
ll=l_nm*(10**(-9))
ph=dati['photons']  #eliminare i punti a 0 --> provato, visto che non cambia nulla, se non un ovvio aumento di 0.0002 K della temperatura. 
#print(ll)
#print(ph)


plt.plot(ll,ph, color= 'green')
plt.title('Spettro stella X')
plt.xlabel("Lunghezza d'onda [m]")
#mettere unità di misura y
plt.show()

plt.hist(ph, bins=50, color='goldenrod', ec='darkgoldenrod',alpha=0.7 )
plt.show()


def B_scatter(l,T,scala):
      """
    Funzione che descrive la densità di energia irradiata da una stella considerando l'effetto dello scattering di Rayleigh. Da utilizzare nel fit
    Parametri
    -----------
        l: Lunghezza d'onda
        T: Temperatura della stella
        Scala: fattore di scala per il fit
    
    Restituisce
        scala*(2*h*(c**2)/((np.exp(h*c/(l*k*T))-1)*(l**5)))*np.exp(-beta(l)*S_teta(pi/4))   
        
    -----------
 
    """
      return scala*(2*h*(c**2)/((np.exp(h*c/(l*k*T))-1)*(l**5)))*np.exp(-beta(l)*S_teta(pi/4))   



pm, pm_cov = optimize.curve_fit(B_scatter,ll,ph,p0=[6000, 1e-13])   #i parametri vanno usati PER FORZA in questo caso, altrimenti non riesce a fare il fit 

Tx=pm[0]
sk=pm[1]


print('{:<40} {:.1f} ± {:.1f} K'.format('La temperatura della StellaX  è:', Tx, np.sqrt(pm_cov[0,0])))


ph_fit=B_scatter(ll,Tx,sk)
#print('valori',ph,ph_fit)


#chi2 =  np.sum(((ph-ph_fit)**2)/ph)    #in alcuni punti divide per zero...
chi2=0
for i in range(len(ph)):
    if ph[i]== 0:
        continue
    chi2+=((ph[i]-ph_fit[i])**2)/ph[i]

# gradi di libertà
d = len(ll)-len(pm)

print('Il chi quadro ridotto vale  ', chi2/d)






fig,axs = plt.subplots(3,1, figsize=(12,6))
fig.subplots_adjust(hspace=0)
axs[0].plot(ll,ph,color='darkgreen', label='Valori osservati')
axs[0].plot(ll,ph_fit, color='red', label='Fit')
axs[0].text(2.5*(10**-6), 300, r'$\chi^2$ rid : {:3.2f} '.format(chi2/d), fontsize=14, color='slategrey')
axs[0].legend(fontsize=14, frameon=False)
axs[1].errorbar(ll, ph/ph_fit,color='royalblue', fmt='o',alpha=0.5, label='Rapporto')
axs[1].grid(True, axis='y')
axs[1].axhline(1, color='red')
axs[1].legend(fontsize=12)
axs[2].errorbar(ll, np.abs(ph_fit-ph),color='violet', fmt='o',alpha=0.5, label='Scarto')
axs[2].grid(True, axis='y')
axs[2].axhline(0, color='red')
axs[2].legend(fontsize=12)
plt.show()






















"""
così come avevo provato al contrario--> probBILMENTE, A PARTE CHE INCONTRAVO ERRORI COMPUTAZIONALI, CON QUALCHE ACCORTEZZA SAREBBE ANDATO BENE UGUALE, AVREI DOVUTO IMPOSTARE PROBABLY DEI PARAMETRI PER FARE IL FIT, MA COMUNQUE, CON IL METODO SOPRA RISULTA TUTTO PIÙ PULITO SENZA ARTEFATTI STRANI....QUI INCOONTRAVO   NaN....



#ph_c=ph*np.exp(beta(l_nm)*S_teta(pi/4))    #il problema  che porta i NaN è lambda , metto in nm
ph_c=ph*np.exp(beta(ll)*S_teta(pi/4))


ph_ccc=np.nan_to_num(ph_c, nan=0) 
mask= ph_ccc <0.01
ph_cf=ph_ccc.copy()
ph_cf[mask]=0


def D_fit(l,T,A):
    return A*2*c*np.exp(-h*c/(l*k*T))/(l**4)

Tx, Tx_cov = optimize.curve_fit(D_fit,ll,ph_cf)

"""


