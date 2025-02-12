import numpy as np
import scipy
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import math
from  scipy import integrate
import argparse
from scipy.constants import c,h,k, pi
import corpo_nero as cn
from corpo_nero import T_sun,T_sau,T_vega,T_rigel,R_T,n_T,N_T,S_z,S_o


dati=pd.read_csv('observed_starX.csv')
#print(dati)
l_nm=dati['lambda (nm)']
ll=l_nm*(10**(-9))
ph=dati['photons']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fit spettro stella ignota')
    parser.add_argument('-m1', '--mod1',    action='store_true', help="Fit prima modalità")
    parser.add_argument('-m2', '--mod2',    action='store_true', help="Fit seconda modalità")
   
    return  parser.parse_args()

args = parse_arguments()


if args.mod1 == True:
      
      plt.plot(ll,ph, color= 'green')
      plt.title('Spettro stella X')
      plt.xlabel("Lunghezza d'onda [m]")
      plt.ylabel(r"radiazione osservata [$J/s m^2$]")
      plt.show()
      plt.show()

      def B_scatter(l,T,scala):
      

            '''
            Funzione che descrive la densità di energia irradiata da una stella considerando l'effetto dello scattering di Rayleigh. Da utilizzare nel fit
            Parametri
            -----------
            l: Lunghezza d'onda
            T: Temperatura della stella
            Scala: fattore di scala per il fit
            
            Restituisce
            
            scala*(2*h*(c**2)/((np.exp(h*c/(l*k*T))-1)*(l**5)))*np.exp(-beta(l)*S_teta(pi/4))   
            
            -----------
            
            '''
            return scala*(2*h*(c**2)/((np.exp(h*c/(l*k*T))-1)*(l**5)))*np.exp(-cn.beta(l)*cn.S_teta(pi/4))   

      pm, pm_cov = optimize.curve_fit(B_scatter,ll,ph,p0=[6000, 1e-13])  

      Tx=pm[0]
      sk=pm[1]


      print('{:<40} {:.1f} ± {:.1f} K'.format('La temperatura della StellaX  è:', Tx, np.sqrt(pm_cov[0,0])))


      ph_fit=B_scatter(ll,Tx,sk)


      #chi2 =  np.sum(((ph-ph_fit)**2)/ph)    #in alcuni punti divide per zero...
      chi2=0
      for i in range(len(ph)):
            if ph[i]== 0:
                  continue
            chi2+=((ph[i]-ph_fit[i])**2)/ph[i]
            
      # gradi di libertà
      d = len(ll)-len(pm)
      
      print('{:<40} {:.2f} '.format('Il chi quadro ridotto vale: ', chi2/d))
            
            
      
            
      fig,axs = plt.subplots(3,1, figsize=(12,6))
      fig.subplots_adjust(hspace=0)
      axs[0].plot(ll,ph,color='darkgreen', label='Valori osservati')
      axs[0].plot(ll,ph_fit, color='red', label='Fit')
      axs[0].text(2.5*(10**-6), 300, r'$\chi^2$ rid : {:3.2f} '.format(chi2/d), fontsize=14, color='slategrey')
      axs[0].legend(fontsize=14, frameon=False)
      axs[0].set_ylabel(r"radiazione  [$J/s m^2$]")
      axs[1].errorbar(ll, ph/ph_fit,color='royalblue', fmt='o',alpha=0.5, label='Rapporto')
      axs[1].grid(True, axis='y')
      axs[1].axhline(1, color='red')
      axs[1].legend(fontsize=12)
      axs[2].errorbar(ll, np.abs(ph_fit-ph),color='violet', fmt='o',alpha=0.5, label='Scarto')
      axs[2].grid(True, axis='y')
      axs[2].axhline(0, color='red')
      axs[2].set_ylabel(r"[$J/s m^2$]")
      axs[2].legend(fontsize=12)
      plt.xlabel("Lunghezza d'onda [m]")
      plt.show()











if args.mod2 == True:
      
      
      
      
      mask= ph>0
      ph_t=ph.copy()
      ll_t=ll.copy()
      ph_t=np.array(ph[mask])
      ll_t=np.array(ll[mask])
      #print(ll_t)
      
      
      
      plt.bar(ll_t,ph_t,width=20*(10**-9), color='gold', edgecolor='goldenrod')
      plt.title('Spettro stella X')
      plt.xlabel("Lunghezza d'onda [m]")
      plt.ylabel(r"radiazione osservata [$J/s m^2$]")
      plt.show()
      
      def spettro_mod(l,T,scala):   
            return scala*(2*h*(c**2)/((np.exp(h*c/(l*k*T))-1)*(l**5)))*np.exp(-cn.beta(l)*cn.S_teta(pi/4))   
      
      
      bins_m=(ll_t[:-1]+ll_t[1:])/2
      #print(bins_m)
      
      params, params_covariance = optimize.curve_fit(spettro_mod,bins_m,ph_t[1:],sigma=np.sqrt(ph_t[1:]),absolute_sigma=True,p0=[5000,10**-10])    
      
      
      Tx=params[0]
      sk=params[1]
      
      
      print('{:<40} {:.1f} ± {:.1f} K'.format('La temperatura della StellaX  è:', Tx, np.sqrt(params_covariance[0,0])))
      
      
      ph_fit=spettro_mod(ll_t,Tx,sk)

      chi2 =  np.sum(((ph_t-ph_fit)**2)/ph_t)
      g=len(bins_m)-len(params)  #gradi di libertà
      print('{:<40} {:.2f} '.format('Il chi quadro ridotto vale: ', chi2/g))



      fig,axs = plt.subplots(3,1, figsize=(12,6))
      fig.subplots_adjust(hspace=0)
      
      axs[0].errorbar(ll_t,ph_fit,yerr=np.sqrt(ph_t),color='red',label='funzione di fit')
      axs[0].bar(ll_t,ph_t,width=20*(10**-9), color='gold', edgecolor='goldenrod', label='spettro fornito')
      axs[0].text(2.5*(10**-6), 300, r'$\chi^2$ rid : {:3.2f} '.format(chi2/g), fontsize=14, color='slategrey')
      axs[0].legend(fontsize=14, frameon=False)
      axs[0].set_ylabel(r"radiazione  [$J/s m^2$]")
      axs[1].errorbar(ll_t,np.abs(ph_fit-ph_t),yerr=np.sqrt(ph_t),color='royalblue', fmt='o',alpha=0.5, label='Scarto')
      axs[1].grid(True, axis='y')
      axs[1].axhline(0, color='red')
      axs[1].legend(fontsize=12)
      axs[1].set_ylabel(r"[$J/s m^2$]")
      axs[2].errorbar(ll_t, np.abs(ph_fit-ph_t)/np.sqrt(ph_t),yerr=np.sqrt(ph_t),color='violet', fmt='o',alpha=0.5, label='Scarto/errore')
      axs[2].grid(True, axis='y')
      axs[2].axhline(0, color='red')
      axs[2].set_ylabel(r"[$J/s m^2$]")
      axs[2].legend(fontsize=12)
      plt.xlabel("Lunghezza d'onda [m]")
      plt.show()



           



