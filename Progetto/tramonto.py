import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from  scipy import integrate
from scipy.constants import c,h,k,g, pi, G
import math


"""
Definizione valori e funzioni utili


il problema che avevo nell'esponente, è che, usando np.exp(arg)-1  ppure np.expm1(arg), veniva calcolato e^... , poi sottratto 1 e poi fatta la divisione. Questo causava overflow nel calcolo dell'esponente, essendo un numero moolto grosso.Dunque, essendo l'argomento dell'esponente >>1, ho considerato direttamente np.exp(-arg) che NON IMPLICA il calcolo di un exp con esponente gigante.  Spero vada bene
"""


T_sun=5.75*(10**3)   #K
T_sau=1.9*(10**3)
T_vega=10*(10**3)
T_rigel=25*(10**3)

#parametri terrestri
R_T=6378000 #m
n_T=1.00029
N_T=2.504*(10**25)
S_z=8000 #m
S_o=np.sqrt((R_T+S_z)**2 -  R_T**2)

#spessore di massa d'aria con angolo generico
def S_teta(t):
    return np.sqrt((R_T*np.cos(t))**2 +  2*R_T*S_z + S_z**2) - R_T*np.cos(t)

#densità di energia irradiata
def B(l,T):
    #return  2*h*(c**2)/(np.expm1(h*c/(l*k*T))*(l**5))   #vera
    return 2*h*(c**2)*np.exp(-h*c/(l*k*T))/(l**5)    #approx


# densità di fotoni per lunghezza d'onda 
def D(l,T):
    #return 2*c/((l**4)*np.expm1(h*c/(l*k*T)))    #vera
    return 2*c*np.exp(-h*c/(l*k*T))/(l**4)     #approx


# probabilità di interazione
def beta(l):
    return 8*pow(np.pi,3)*pow(n_T**2 - 1,2)/(3*N_T*l**4)


def D_scatter(l,s,T):
    return  D(l,T)*np.exp(-beta(l)*s)


la=np.random.uniform(low=10**(-8), high=5*10**(-6), size=10000)  # in  m   considero spettro da UV a infrarossi
#la=np.random.uniform(low=1, high=5000, size=10000)
lmbd=np.sort(la)






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
#PLOT NORMALI SOLE
plt.plot(lmbd,B(lmbd,T_sun), color='mediumvioletred')
plt.show()

plt.scatter(lmbd,D(lmbd,T_sun), color='darkgreen', marker='2')
plt.scatter(lmbd,D_scatter(lmbd,S_z,T_sun),  color='gold', marker= '2')
plt.scatter(lmbd,D_scatter(lmbd,S_o,T_sun),  color='purple', marker= '2')
plt.show()




DA FARE:
far vedere distriuzione lambda
provare a fare i plot senza scatter, ordinando i punti
individuare il picco
dividere per 'colori'
"""






"""
per il flusso integrato integro D(l,T) tra i limiti scelti di lambda, mi va in overflow !!!
"""

te=np.random.uniform(low=0,high=pi,size=1000)
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
axs[0,1].plot(S_teta(teta),w, color='pink')
axs[1,0].plot(S_teta(teta),v, color='violet')
axs[1,1].plot(S_teta(teta),r, color='sienna')
plt.show()



"""
GRAFICO NORMALE SOLE
plt.scatter(S_teta(teta),f, color='teal')
plt.show()




FINO A QUI OKK    (considerando buona l'approssimazione dell'esponenziale)
DA FARE
fare vedere distribuzione teta e lmbd
fare meglio i grafici e spiegare andamenti
"""




























"""
STELLA X

PROBABILMENTE NON TORNA PERCHÈ (e in questo caso dovrei modificare anche la roba prima) sto facendo il fit con la densità di fotonii, e io qua ho il numero di fotoni, quindi...

cose da fare: scrivere bene il risultato, confrontare con il grafico fittato, far vedere lo scarto, chi quadro


"""




dati=pd.read_csv('observed_starX.csv')
#print(dati)
l_nm=dati['lambda (nm)']
ll=l_nm*(10**(-9))
ph=dati['photons']
#print(ll)
#print(ph)


plt.plot(ll,ph, color= 'green')
plt.show()


ph_c=ph*np.exp(beta(ll)*S_teta(pi/4))  #in nm, sennò non calcola

#porco=[]
#for i in range(len(ph)):
 #   porco.append(ph[i]-ph_c[i])

    
#print(porco)



#print(ph_c)
Tx, Tx_cov = optimize.curve_fit(D,ll,ph_c)

print(Tx)
print(Tx_cov)

plt.plot(ll,D(ll,Tx),color='red')
plt.plot(ll,ph_c, color= 'blue')
plt.show()


