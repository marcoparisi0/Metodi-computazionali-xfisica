import numpy as np
from scipy.constants import c,h,k, pi
import math



# VALORI

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



#DEFINIZIONE FUNZIONI

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
