import sys
import numpy as np
import scipy 
import matplotlib.pyplot as plt
from matplotlib import  transforms

#print(sys.path)
sys.path.append('/home/muuone/MCF/Metodi-computazionali-xfisica/E10')

def random_walk2d(step, N):
   
    # Array posizioni x e y (partenza dalla posizione 0,0)
    deltax = np.array([0])
    deltay = np.array([0])

    # Valori random di phi per gli N passi distribuiti uniformemente
    fi = np.random.uniform(low=0, high=2*np.pi, size=N)

    # Ciclo sui valori di phi per calcolare gli spostamenti 
    for p in fi:

        # Valori temporanei con nuovo step
        tmpx = deltax[-1] + step*np.cos(p)    #delta[-1]   prende il valore preecedente
        tmpy = deltay[-1] + step*np.sin(p)

        # Appendo nuove posizione agli array degli spostamenti
        deltax = np.append(deltax, tmpx)
        deltay = np.append(deltay, tmpy)
        
    return deltax, deltay

def p(phi):
    return (np.sin(phi/2))/4

def phhi(N):
    s=np.random.random(N) #valore random distr uniformemente per cumulativa
    phi=2*np.arccos(1-2*s)
    return phi

#plt.hist(phhi(1000),bins=25, alpha=0.8, color='royalblue', ec='darkblue')
#plt.show()
    
def rw2d_asimm(step,N):
  
    deltax = np.array([0])
    deltay = np.array([0])

    for p in phhi(N):

        tmpx = deltax[-1] + step*np.cos(p)    #delta[-1]   prende il valore preecedente
        tmpy = deltay[-1] + step*np.sin(p)

        deltax = np.append(deltax, tmpx)
        deltay = np.append(deltay, tmpy)
        
    return deltax, deltay


def asimmdu(step,sf):
    
    deltax = np.array([0])
    deltay = np.array([0])
   # passo=step
    while(deltax[-1]<=200*sf):
        #passo+=sf
        p = np.random.uniform(low=0, high=2*np.pi)

       # tmpx = deltax[-1] + passo*np.cos(p)
        tmpx=deltax[-1] + step*(np.cos(p) + sf)
        tmpy = deltay[-1] + step*np.sin(p)

        deltax = np.append(deltax, tmpx)
        deltay = np.append(deltay, tmpy)
        
    return deltax, deltay
