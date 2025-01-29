import sys
import numpy as np
import scipy 
import matplotlib.pyplot as plt
from matplotlib import  transforms
import rw


N1= 1000
step1 =2

plt.subplots(figsize=(9,8))
for i in range(5):
    x1,y1 = rw.rw2d_asimm(step1, N1)
    plt.plot(x1,y1)
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$\Delta y$')
plt.show()


passo= 1
sf1 =0.1
sf2=0.01


fig, axs = plt.subplots(1,2,figsize=(12,6))
for i in range(5):
    x1,y1 = rw.asimmdu(passo, sf1)
    x2,y2=rw.asimmdu(passo, sf2)
    axs[0].plot(x1,y1)
    axs[1].plot(x2,y2)
axs[0].set_xlabel(r'$\Delta x$')
axs[0].set_ylabel(r'$\Delta y$')
axs[1].set_xlabel(r'$\Delta x$')
axs[1].set_ylabel(r'$\Delta y$')
plt.show()


