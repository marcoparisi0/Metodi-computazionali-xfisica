
import sys
import numpy as np
import scipy 
import matplotlib.pyplot as plt
from matplotlib import  transforms
import rw


"""

N1= 1000
step1 =1

plt.subplots(figsize=(9,8))
for i in range(100):
    x1,y1 = rw.random_walk2d(step1, N1)
    plt.plot(x1,y1)
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$\Delta y$')
plt.show()


N2= 100
step2 =1


plt.subplots(figsize=(9,8))
for i in range(100):
    x2,y2 = rw.random_walk2d(step2, N2)
    plt.plot(x2,y2)
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$\Delta y$')
plt.show()


N3= 1000
step3 =2

plt.subplots(figsize=(9,8))
for i in range(100):
    x3,y3 = rw.random_walk2d(step3, N3)
    plt.plot(x3,y3)
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$\Delta y$')
plt.show()
"""

N0= 1000
step0 =2

fig, axs = plt.subplots(1,2,figsize=(12,6))
for i in range(5):
    x,y = rw.random_walk2d(step0, N0)
    axs[0].plot(x,y)
    axs[1].plot(range(0,N0+ 1), x**2+y**2)  #range(start,stop,step) genera array di numeri interi)
axs[0].set_xlabel(r'$\Delta x$')
axs[0].set_ylabel(r'$\Delta y$')
axs[1].set_xlabel('num. passi')
axs[1].set_ylabel('d^2')
plt.show()
