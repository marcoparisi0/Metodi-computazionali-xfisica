import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
import math
from  scipy import integrate
import matplotlib.colors as mcolors
from scipy.constants import c,h,k, pi
import argparse
import corpo_nero as cn
from corpo_nero import T_sun,T_sau,T_vega,T_rigel,R_T,n_T,N_T,S_z,S_o



def parse_arguments():
    parser = argparse.ArgumentParser(description='Analisi spettri di emissione stelle')
    parser.add_argument('-s', '--Sun',    action='store_true', help="Mostra l'analisi completa relativa al Sole")
    parser.add_argument('-sa', '--Saurigae',    action='store_true', help="Mostra l'analisi completa relativa alla stella Saurigae")
    parser.add_argument('-v', '--Vega',    action='store_true', help="Mostra l'analisi completa relativa alla stella Vega")
    parser.add_argument('-r', '--Rigel',    action='store_true', help="Mostra l'analisi completa relativa alla stella Rigel")
    return  parser.parse_args()

args = parse_arguments()





def analisi_stella(nome,T,ppo,ppz):
    """
    T: temperatura
    ppo: parametro per trovare il massimo, stella ad orizzonte
    ppz: parametro per trovare il massimo, stella allo zenith

    """

    la=np.random.uniform(low=10**(-8), high=5*10**(-6), size=10000)
    
    y0=np.random.uniform(low=0,high=np.max(cn.D(la,T)),size=10000)
    hit0= y0 <= cn.D(la,T)
    lmbd0=la[hit0]
    plt.hist(la,bins=100,color='pink', label=r'valori $\lambda$ generati')
    plt.hist(lmbd0,bins=100, color='lightgreen',label=r'valori $\lambda$ selezionati')
    plt.title('Distribuzione fotoni emessi da: {}'.format(nome))
    plt.legend()
    plt.show()

    y_or=np.random.uniform(low=0,high=np.max(cn.D_scatter(la,S_o,T)),size=10000)
    hit_or= y_or <= cn.D_scatter(la,S_o,T)
    lmbd_or=la[hit_or]
    plt.hist(la,bins=100,color='pink', label=r'valori $\lambda$ generati')
    plt.hist(lmbd_or,bins=100, color='darkorchid',label=r'valori $\lambda$ selezionati')
    plt.legend()
    plt.title('Distribuzione fotoni con scattering ad orizzonte, emessi da: {}'.format(nome))
    plt.show()

    y_z=np.random.uniform(low=0,high=np.max(cn.D_scatter(la,S_z,T)),size=10000)
    hit_z= y_z <= cn.D_scatter(la,S_z,T)
    lmbd_z=la[hit_z]
    plt.hist(la,bins=100,color='pink', label=r'valori $\lambda$ generati')
    plt.hist(lmbd_z,bins=100, color='teal',label=r'valori $\lambda$ selezionati')
    plt.legend()
    plt.title('Distribuzione fotoni con scattering allo Zenith, emessi da: {}'.format(nome))
    plt.show()

    #confronto
    plt.hist(lmbd0,bins=100, color='lightgreen',alpha=0.5,label=r'valori $\lambda$ no scattering')
    plt.hist(lmbd_or,bins=100, color='darkorchid',alpha=0.5,label=r'valori $\lambda$ scattering orizzonte')
    plt.hist(lmbd_z,bins=100, color='teal',alpha=0.5,label=r'valori $\lambda$ scattering zenith')
    plt.legend()
    plt.title('Confronto tra le tre distribuzioni , {}'.format(nome))
    plt.show()
    



    


    """
    Studio più dettagliato con lambda distribuiti uniformemente
    
    
    lmbd=np.sort(la)
    # Filtro per  la parte visibile dello spettro per creare la parte colorata nel grafico
    mask = (lmbd >= 380e-9) & (lmbd <= 750e-9)
    lmbd_vis = lmbd[mask]
    # Creazione della colormap 
    cmap = plt.get_cmap("rainbow")  
    norm = mcolors.Normalize(vmin=380e-9, vmax=750e-9)  # Normalizzazione solo per il visibile
    colors = cmap(norm(lmbd_vis))


    
    te=np.random.uniform(low=0,high=pi,size=1000)  #angoli, mi serviranno per l'analisi con distanza variabile della stella dallo Zenith

    plt.hist(te, bins=25, alpha=0.8, color='violet', ec='darkviolet')
    plt.title(r'Distribuzione uniforme angoli [0,$\pi$]')
    plt.show()

    teta=np.sort(te)



    
    E=cn.B(lmbd,T)
    plt.plot(lmbd,E, color='black')
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.title('Spettro di emissione {}, no scattering'.format(nome))
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"radiazione emessa [$J/s m^2$]")


    E_vis = E[mask]


    for i in range(len(lmbd_vis) - 1):
        plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [E_vis[i], E_vis[i + 1]], color=colors[i])
    plt.show()


    

    #DISTRIBUZIONE FOTONI SENZA ASSORBIMENTO
    plt.plot(lmbd,cn.D(lmbd,T), color='orchid')
    plt.title('distribuzione fotoni {} senza assorbimento'.format(nome))
    plt.xlabel(r'$\lambda$ [m]')
    plt.ylabel(r'$fotoni/sm^3$')
    plt.show()
    



    #DISTRIBUZIONE FOTONI  ORIZZONTE
    Tr=cn.D_scatter(lmbd,S_o,T)
    Tr_vis = Tr[mask]

    def D_scatter_min_o(l):
       return -cn.D_scatter(l,S_o,T)
    ris_o=optimize.minimize(D_scatter_min_o,x0=ppo)
    l_max_o=ris_o.x[0]
    o_max=cn.D_scatter(l_max_o,S_o,T)


    plt.plot(lmbd,Tr, color='black')
    plt.plot(l_max_o,o_max,'o',color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_o))
    plt.plot([l_max_o, l_max_o], [0, o_max], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering {}  ad orizzonte'.format(nome))
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")

    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Tr_vis[i], Tr_vis[i + 1]], color=colors[i])
    plt.show()





    # ZENITH
    Z=cn.D_scatter(lmbd,S_z,T)
    Z_vis = Z[mask]

    def D_scatter_min_z(l):
       return -cn.D_scatter(l,S_z,T)
    ris_z=optimize.minimize(D_scatter_min_z,x0=ppz)
    l_max_z=ris_z.x[0]
    z_max=cn.D_scatter(l_max_z,S_z,T)

    plt.plot(lmbd,Z, color='black')
    plt.plot(l_max_z,z_max,'o', color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_z))
    plt.plot([l_max_z, l_max_z], [0, z_max], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering {} allo zenith'.format(nome))
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")
    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Z_vis[i], Z_vis[i + 1]], color=colors[i])

    plt.show()



    #confronto
    plt.plot(lmbd,cn.D(lmbd,T), color='goldenrod',label='senza assorbimento')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_z,T),  color='gold',label='scattering posizione Zenith')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_o,T),  color='tan',label='scattering posizione Orizzonte')
    plt.legend()
    plt.title('Confronto distribuzioni fotoni')
    plt.xlabel(r'$\lambda$ [m]')
    plt.ylabel(r'$fotoni/sm^3$')
    plt.show()


    #andamento flusso fotoni in funzione della posizione della stella
    
    f=[]
    for t in teta:
       integranda=cn.D_scatter(lmbd,cn.S_teta(t),T)
       flusso=integrate.simpson(integranda, x=lmbd)
       f.append(flusso)
    plt.plot(cn.S_teta(teta),f, color='teal')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title('Andamento flusso integrato di fotoni')
    plt.xlabel('Distanza  {}  dallo Zenith [m]'.format(nome))
    plt.ylabel(r'$fotoni/s m^2$')
    plt.show()
    """







"""
-------------------------------------------SOLE---------------------------------------------------------------------
"""

if args.Sun == True:
    analisi_stella("Sole",T_sun,8*(10**-7),5*(10**-7))



"""
-------------------------------------------SAURIGAE---------------------------------------------------------------------
"""
if args.Saurigae == True:
    analisi_stella("Saurigae",T_sau,10**-6,10**-6)




"""
-------------------------------------------VEGA---------------------------------------------------------------------
"""
if args.Vega == True:
    analisi_stella("Vega",T_vega,10**-6,4*(10**-7))

    



"""
-------------------------------------------RIGEL---------------------------------------------------------------------
"""
if args.Rigel == True:
    analisi_stella("Rigel",T_rigel,10**-6,4*(10**-7))
