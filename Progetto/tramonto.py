import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from  scipy import integrate
from scipy.constants import c,h,k,e
import math

#prima richiesta, devo usare D(lambda,T)

T_sun=5.75*(10**3)


def D(l):
    return 2*c/((l**4)*(pow(e,h*c/l*k*T_sun) -1))


lmbd=np.random.uniform(low=10**(-8), high=10**(-6), size=1000)  # considero spettro da UV a infrarossi


plt.plot(lmbd,D(lmbd))
plt.show()
