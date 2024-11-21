import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  scipy import integrate
import math
import argparse






def parse_arguments():
    parser = argparse.ArgumentParser(description='Esempio utilizzo argarse.')    
    parser.add_argument('-V2', '--potenziale2',    action='store_true',      help='grafico potenziale-x0')  
    parser.add_argument('-V6', '--potenziale6',    action='store_true',  help='grafico potenziale-x0')
    parser.add_argument('-V4', '--potenziale4',    action='store_true',  help='grafico potenziale-x0')
    return  parser.parse_args()








x0 =float( input('Immettere punto di partenza da cui calcolare il periodo (intervallo di integrazione [0, x0] : '))
k=float(input('inserire la costante elastica k:'))
m=float(input('inserire la massa:'))


def V(x):
    return k*(x**6)

x  = np.arange(0,x0, 0.1)
for b in x:
    if x0==b:
        continue
integranda=np.sqrt(8*m)*(1/(np.sqrt(V(x0)-V(x))))
integrale=integrate.simpson(integranda, dx=x0/10000)

print('il valore del periodo per kx^6è   T:', integrale)


"""
da qui provo a creare un grafico che mi dia l'andamento di V al variare del punto di partenza
"""
x0_0=np.arange(0,100,0.5)

c=[]
for i in x0_0:
    for p in x:
       if i==p:
           continue
    integranda_0=np.sqrt(8*m)*(1/(np.sqrt(V(i)-V(x))))
    cucu=integrate.simpson(integranda_0,dx = 0.01)
    c.append(cucu)


"""
rifaccio l'analisi per potenziale kx^2
"""


xi =float( input('Immettere punto di partenza da cui calcolare il periodo (intervallo di integrazione [0, xi] : '))
j=float(input('inserire la costante elastica j:'))
M=float(input('inserire la massa:'))

def V_star(x):
    return j*(x**2)

ics  = np.arange(0,x0, 0.1)
for b in ics:
    if xi==b:
        continue
intda=np.sqrt(8*M)*(1/(np.sqrt(V_star(xi)-V_star(ics))))
intle=integrate.simpson(intda, dx=xi/10000)

print('il valore del periodo per kx^2 è   T:', intle)


xi_i=np.arange(0,100,0.5)

d=[]
for i in xi_i:
    for p in ics:
       if i==p:
           continue
    intda_i=np.sqrt(8*M)*(1/(np.sqrt(V_star(i)-V_star(ics))))
    miu=integrate.simpson(intda_i,dx = 0.01)
    d.append(miu)


fig,ax = plt.subplots(1,2, figsize=(12,6) )


ax[0].plot(x0_0, c, color = 'blue')
ax[0].set_xlabel('x0')
ax[0].set_ylabel('Potenziale  kx^6')
ax[0].set_title('andamento di V al variare del punto di partenza x0')

ax[1].plot(xi_i, d, color = 'orange')
ax[1].set_xlabel('x0')
ax[1].set_ylabel('Potenziale kx^2')
ax[1].set_title('andamento di V al variare del punto di partenza x0')
plt.show()



"""
rifaccio l'analisi per potenziale kx^4
"""


x4 =float( input('Immettere punto di partenza da cui calcolare il periodo (intervallo di integrazione [0, xi] : '))
K=float(input('inserire la costante elastica K:'))
ma=float(input('inserire la massa:'))

def V_4(x):
    return K*(x**4)

xx  = np.arange(0,x4, 0.1)
for b in xx:
    if x4==b:
        continue
intda4=np.sqrt(8*ma)*(1/(np.sqrt(V_4(x4)-V_4(xx))))
intle4=integrate.simpson(intda4, dx=x4/10000)

print('il valore del periodo per kx^2 è   T:', intle4)


xi_4=np.arange(0,100,0.5)

e=[]
for i in xi_4:
    for p in xx:
       if i==p:
           continue
    intda_4=np.sqrt(8*ma)*(1/(np.sqrt(V_4(i)-V_4(xx))))
    bau=integrate.simpson(intda_4,dx = 0.01)
    e.append(bau)




    



def main():
    args = parse_arguments()

    
    if args.potenziale6 == True:
        plt.plot(x0_0, c, color = 'blue')
        plt.xlabel('x0')
        plt.ylabel('Potenziale  kx^6')
        plt.title('andamento di V al variare del punto di partenza x0')
        plt.show()

    if args.potenziale2 == True:
        plt.plot(xi_i, d, color = 'orange')
        plt.xlabel('x0')
        plt.ylabel('Potenziale kx^2')
        plt.title('andamento di V al variare del punto di partenza x0')
        plt.show()

    if args.potenziale4 == True:
        plt.plot(xi_4, e, color = 'orange')
        plt.xlabel('x0')
        plt.ylabel('Potenziale kx^4')
        plt.title('andamento di V al variare del punto di partenza x0')
        plt.show()
        
if __name__ == '__main__':
    main()



    
