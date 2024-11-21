import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



dataset = pd.read_csv('http://opendata.cern.ch/record/5203/files/Jpsimumu.csv')
E1=dataset['E1']
E2=dataset['E2']
px1=dataset['px1']
py1=dataset['py1']
pz1=dataset['pz1']
px2=dataset['px2']
py2=dataset['py2']
pz2=dataset['pz2']

m_inv=np.sqrt((E1+E2)**2 - ((px1+px2)**2 +(py1+py2)**2 + (pz1+pz2)**2))



plt.hist (m_inv , bins=100, color= 'red')
plt.title('istogramma massainvariante')
plt.show()

n,bins,p=plt.hist(m_inv , bins=100, range=(2.8,3.5), color= 'red')
"""
scrivendo così, n mi da il numero di elementi per ogni bin, e bins mi da la x di ogni inizio intervallo
"""
plt.title('istogramma massainvariante ristretto al picco')
plt.show()




def f_g1(x,A,m,sigma,p1,p0):
    cucu=A*np.exp(-((x-m)**2)/(2*sigma**2)) + p1*x +p0
    return cucu

def f_g2(x, A1, A2, sigma1, sigma2, m, p1, p0):

    gauss1 = A1 * np.exp(-((x - m)**2) / (2 * sigma1**2))
    gauss2 = A2 * np.exp(-((x - m)**2) / (2 * sigma2**2))
    lineare = p1 * x + p0
    return gauss1 + gauss2 + lineare


bins_m=(bins[:-1]+bins[1:])/2
"""
per la frequenza mi serve il centro del bin 
[:-1] mi prende tutti gli elementi dell'array meno l'ultimo
[1:] prende tutti i valori meno il primo
questo perchè nei bin, ogni valore corrisponde sia all'inizio dell iesimo intervallo che alla fine dell i-1esimo,  mente il rimo e l'ultimo sono esclusivamente inizio e fine
devo semplicemente fare la media tra due array.
"""



params, params_covariance = optimize.curve_fit(f_g1,bins_m,n,sigma=np.sqrt(n),absolute_sigma=True,p0=[1,1,1,1,1])  #l'errore è radice di n perchè è una poissoniana
yfit=f_g1(bins_m,params[0],params[1],params[2],params[3],params[4])
for s in params:
    print('params  = {:<10}'.format(s))
for row in params_covariance:
    for s in row:
        print('params_cov  = {:<10}'.format(s))
chi2 =  np.sum( (yfit - n)**2 /n ) 
g = len(bins_m)-len(params)  #gradi di libertà
print('il chi ridotto è= {:.4f}'.format(chi2/g))




paramse, paramse_covariance = optimize.curve_fit(f_g2,bins_m,n,sigma=np.sqrt(n),absolute_sigma=True,p0=[1,1,1,1,1,1,1])
yfitte=f_g2(bins_m,paramse[0],paramse[1],paramse[2],paramse[3],paramse[4],paramse[5],paramse[6])
for s in paramse:
    print('params new = {:<10} '.format(s))
for row in paramse_covariance:
    for s in row:
        print('params_cov new = {:<10} '.format(s))

chi2_2 =  np.sum( (yfitte - n)**2 /n ) 
g_2 = len(bins_m)-len(paramse)  #gradi di libertà
print('il chi ridotto con fit migliore  è=  {:.4f}'.format(chi2_2/g_2))





fig, axs = plt.subplots(3,2,figsize=(12,6))
axs[0,0].hist(m_inv,bins=100, range=(2.8,3.5), color= 'red',label='esperimento',alpha=0.3)
axs[0,0].plot(bins_m,yfit,color='red',label='fit')
axs[0,0].set_title('massa invariante')
axs[0,0].legend(fontsize=8)
axs[1,0].errorbar(bins_m,np.abs(n-yfit),yerr=np.sqrt(n),fmt='o', color='blue')
axs[1,0].set_title('scarti tra misurazioni e fit')
axs[2,0].errorbar(bins_m,np.abs(n-yfit)/(np.sqrt(n)),yerr=np.sqrt(n),fmt='o',color='blue')
axs[2,0].set_title('scarti tra misurazioni e fit diviso errore')
axs[0,1].hist(m_inv,bins=100, range=(2.8,3.5), color= 'orange',label='esperimento',alpha=0.3)
axs[0,1].plot(bins_m,yfitte,color='purple',label='fit2')
axs[0,1].set_title('massa invariante')
axs[0,1].legend(fontsize=8)
axs[1,1].errorbar(bins_m,np.abs(n-yfitte),yerr=np.sqrt(n),fmt='o', color='green')
axs[1,1].set_title('scarti tra misurazioni e fit più preciso')
axs[2,1].errorbar(bins_m,np.abs(n-yfitte)/(np.sqrt(n)),yerr=np.sqrt(n),fmt='o',color='green')
axs[2,1].set_title('scarti tra misurazioni e fit diviso errore')
axs[2,0].grid(True, axis='y')
axs[2,1].grid(True, axis='y')
axs[1,0].grid(True, axis='y')
axs[1,1].grid(True, axis='y')
plt.show()



