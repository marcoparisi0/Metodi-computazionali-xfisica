import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from  scipy import integrate
from scipy.constants import c,h,k,g, pi, G
import math



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


def S_teta(t):
    return np.sqrt((R_T*np.cos(t))**2 +  2*R_T*S_z + S_z**2) - R_T*np.cos(t)


def B(l,T):
   return  2*h*(c**2)/(np.expm1(h*c/(l*k*T))*(l**5))

def D(l,T):
    return 2*c/((l**4)*np.expm1(h*c/(l*k*T)))


def beta(l):
    return 8*pow(np.pi,3)*pow(n_T**2 - 1,2)/(3*N_T*l**4)


def D_scatter(l,s):
    return  D(l)*np.exp(-beta(l)*s)




lmbd=np.random.uniform(low=10**(-8), high=5*10**(-6), size=10000)  # in  m   considero spettro da UV a infrarossi



plt.scatter(lmbd, B(lmbd,T_sun), color='mediumvioletred')
plt.show()

plt.scatter(lmbd,D(lmbd,T_sun), color='darkgreen', marker='2')
plt.scatter(lmbd,D_scatter(lmbd,S_z),  color='gold', marker= '2')
plt.scatter(lmbd,D_scatter(lmbd,S_o),  color='purple', marker= '2')
plt.show()

