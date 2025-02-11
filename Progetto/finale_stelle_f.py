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
    parser.add_argument('-s', '--Sun',    action='store_true', help="Mostra l'analisi  relativa al Sole")
    parser.add_argument('-s2', '--Sun2',    action='store_true', help="Mostra un'analisi differente  relativa al Sole")
    parser.add_argument('-sa', '--Saurigae',    action='store_true', help="Mostra l'analisi  relativa alla stella Saurigae")
    parser.add_argument('-sa2', '--Saurigae2',    action='store_true', help="Mostra un'analisi differente relativa alla stella Saurigae")
    parser.add_argument('-v', '--Vega',    action='store_true', help="Mostra l'analisi  relativa alla stella Vega")
    parser.add_argument('-v2', '--Vega2',    action='store_true', help="Mostra un'analisi differente relativa alla stella Vega")
    parser.add_argument('-r', '--Rigel',    action='store_true', help="Mostra l'analisi  relativa alla stella Rigel")
    parser.add_argument('-r2', '--Rigel2',    action='store_true', help="Mostra un'analisi differente  relativa alla stella Rigel")
    return  parser.parse_args()

args = parse_arguments()




def analisi_stella(nome,T,mi,ma):
    """
    T: Temperatura stella
    mi: minimo lunghezza d'onda considerata
    ma: massimo lunghezza d'onda considerata
    """
    
    """
    Non potendo ottenere analiticamente l'integrale di D(l,T) avevo provato a fare senza successo:

    num_foto=1000    #numero fotoni simulati
    la=np.linspace(1e-9, 5000e-9, 1000)
    spettro=cn.D(la,T)
    I_tot=integrate.simpson(spettro,x=la)
    yi=np.random.random(num_foto)
    print(yi)

    def c(l):
        lh=np.linspace(1e-9,l)
        return integrate.simpson(cn.D(lh, T),x=lh)/I_tot

    lmbda=[]
    for i in yi:
        lmbda.append(1.0/c(i))
    lmbd0=np.array(lmbda)
    lmbd=np.sort(lmbd0)
    """

    
    la=np.random.uniform(low=mi, high=ma, size=10000)
    
    y0=np.random.uniform(low=0,high=np.max(cn.D(la,T)),size=10000)
    hit0= y0 <= cn.D(la,T)
    lmbd0=la[hit0]
    plt.hist(la,bins=100,color='pink', label=r'valori $\lambda$ generati')
    n0,bins0,_0=plt.hist(lmbd0,bins=100, color='orangered',label=r'valori $\lambda$ selezionati')
    plt.title('Distribuzione fotoni emessi da: {}'.format(nome))
    plt.xlabel("Lunghezza d'onda [m]")
    plt.legend()
    plt.show()

    
    y_or=np.random.uniform(low=0,high=np.max(cn.D_scatter(la,S_o,T)),size=10000)
    hit_or= y_or <= cn.D_scatter(la,S_o,T)
    lmbd_or=la[hit_or]
    plt.hist(la,bins=100,color='pink', label=r'valori $\lambda$ generati')
    n_or,bins_or,_or=plt.hist(lmbd_or,bins=100, color='darkorchid',label=r'valori $\lambda$ selezionati')
    plt.legend()
    plt.title('Distribuzione fotoni con scattering ad orizzonte, emessi da: {}'.format(nome))
    plt.xlabel("Lunghezza d'onda [m]")
    plt.show()
    

    y_z=np.random.uniform(low=0,high=np.max(cn.D_scatter(la,S_z,T)),size=10000)
    hit_z= y_z <= cn.D_scatter(la,S_z,T)
    lmbd_z=la[hit_z]
    plt.hist(la,bins=100,color='pink', label=r'valori $\lambda$ generati')
    n_z,bins_z,_z=plt.hist(lmbd_z,bins=100, color='teal',label=r'valori $\lambda$ selezionati')
    plt.legend()
    plt.title('Distribuzione fotoni con scattering allo Zenith, emessi da: {}'.format(nome))
    plt.xlabel("Lunghezza d'onda [m]")
    plt.show()

    #confronto
    plt.hist(lmbd0,bins=100, color='orangered',alpha=0.5,label=r'valori $\lambda$ no scattering')
    plt.hist(lmbd_or,bins=100, color='darkorchid',alpha=0.5,label=r'valori $\lambda$ scattering orizzonte')
    plt.hist(lmbd_z,bins=100, color='teal',alpha=0.5,label=r'valori $\lambda$ scattering zenith')
    plt.legend()
    plt.title('{}, confronto tra le tre distribuzioni, T = {} K'.format(nome,T))
    plt.xlabel("Lunghezza d'onda [m]")
    plt.ylabel("Distribuzione fotoni")
    plt.show()


    i0=np.argmax(n0) #così trovo il bin più frequente
    m0=(bins0[i0]+bins0[i0+1])/2
    print(r'{}:{} =  {:.2f} $\pm$ {:.2f} nm '.format(nome,r"il valore più frequente senza scattering corrisponde alla $\lambda$",m0*10**9,((bins0[1]-bins0[0])/np.sqrt(12))*10**9))

    i_or=np.argmax(n_or) 
    m_or=(bins_or[i_or]+bins_or[i_or+1])/2
    print(r'{}:{} =  {:.2f} $\pm$ {:.2f} nm'.format(nome,r"il valore più frequente con scattering ad orizzonte corrisponde alla $\lambda$",m_or*10**9,((bins_or[1]-bins_or[0])/np.sqrt(12))*10**9))

    i_z=np.argmax(n_z) 
    m_z=(bins_z[i_z]+bins_z[i_z+1])/2
    print(r'{}:{} =  {:.2f} $\pm$ {:.2f} nm'.format(nome,r"il valore più frequente con scattering allo zenith corrisponde alla $\lambda$ ",m_z*10**9,((bins_z[1]-bins_z[0])/np.sqrt(12))*10**9))


    #andamento flusso fotoni in funzione della posizione della stella
    #utilizzo distribuzione uniforme di angoli, e integro nelle lunghezze d'onda usando il metodo montecarlo
    teta=np.random.uniform(low=0,high=pi/2,size=1000)
    integrale=[]
    for t in teta:
        integrale.append((ma-mi)*np.sum(cn.D_scatter(la,cn.S_teta(t),T))/10000)
    plt.scatter(teta*180/pi,integrale,color='lightcoral',alpha=0.6)
    plt.xlabel(r"posizione {} dallo zenith [gradi]".format(nome))
    plt.ylabel(r"flusso fotoni  $fotoni/s m^2$")
    plt.show()













    

def analisi_differente_stella(nome,T,mi,ma,pp,ppo,ppz):
    """
    T: temperatura
    mi: minimo lunghezza d'onda considerata
    ma: massimo lunghezza d'onda considerata
    pp: parametro per trovare il massimo, no scattering 
    ppo: parametro per trovare il massimo, stella ad orizzonte
    ppz: parametro per trovare il massimo, stella allo zenith

    

    Studio differente con lambda distribuiti uniformemente e integrazione con scipy simpson
    """
    la=np.random.uniform(low=mi, high=ma, size=10000)
    
    lmbd=np.sort(la)
    # Filtro per  la parte visibile dello spettro per creare la parte colorata nel grafico
    mask = (lmbd >= 380e-9) & (lmbd <= 750e-9)
    lmbd_vis = lmbd[mask]
    # Creazione della colormap 
    cmap = plt.get_cmap("rainbow")  
    norm = mcolors.Normalize(vmin=380e-9, vmax=750e-9)  # Normalizzazione solo per il visibile
    colors = cmap(norm(lmbd_vis))


    
    te=np.random.uniform(low=0,high=pi/2,size=1000)  #angoli, mi serviranno per l'analisi con distanza variabile della stella dallo Zenith

    plt.hist(te, bins=25, alpha=0.8, color='violet', ec='darkviolet')
    plt.title(r'Distribuzione uniforme angoli [0,$\pi$/2]')
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
    A=cn.D(lmbd,T)
    A_vis=A[mask]

    def D_scat(l):
        return -cn.D(l,T)
    ris=optimize.minimize(D_scat,x0=ppz)
    l_max=ris.x[0]
    mmm=cn.D(l_max,T)

    
    plt.plot(lmbd,A, color='black')
    plt.plot(l_max,mmm,'o',color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max))
    plt.plot([l_max, l_max], [0,mmm], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.title('distribuzione fotoni {} senza assorbimento'.format(nome))
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.xlabel(r'$\lambda$ [m]')
    plt.ylabel(r'$fotoni/sm^3$')

    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [A_vis[i], A_vis[i + 1]], color=colors[i])
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
    ris_z=optimize.minimize(D_scatter_min_z,x0=pp)
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
    plt.plot(lmbd,cn.D(lmbd,T), color='orangered',label='senza assorbimento')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_z,T),  color='steelblue',label='scattering posizione Zenith')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_o,T),  color='tan',label='scattering posizione Orizzonte')
    plt.legend()
    plt.title('Confronto distribuzioni fotoni')
    plt.xlabel(r'$\lambda$ [m]')
    plt.ylabel(r'$fotoni/sm^3$')
    plt.show()


    #andamento flusso fotoni in funzione della posizione della stella
    #integro con scipy
    
    f=[]
    for t in teta:
       integranda=cn.D_scatter(lmbd,cn.S_teta(t),T)
       flusso=integrate.simpson(integranda, x=lmbd)
       f.append(flusso)
    plt.plot(cn.S_teta(teta),f, color='teal')
    plt.title('Andamento flusso integrato di fotoni')
    plt.xlabel(r"Spessore massa d' aria incontrata dai raggi di {}  [m]".format(nome))
    plt.ylabel(r'$fotoni/s m^2$')
    plt.show()
    











"""
-------------------------------------------SOLE---------------------------------------------------------------------
"""
if args.Sun == True:
    analisi_stella("Sole",T_sun,10**-8,5*10**-6)
    
if args.Sun2 == True:
    analisi_differente_stella("Sole",T_sun,10**-8,5*10**-6, 5*(10**-7), 8*(10**-7), 5*(10**-7))



"""
-------------------------------------------SAURIGAE---------------------------------------------------------------------
"""
if args.Saurigae == True:
    analisi_stella("Saurigae",T_sau,10**-8,10**-5)
    
if args.Saurigae2 == True:
    analisi_differente_stella("Saurigae",T_sau,10**-8, 10**-5, 1.3*(10**-6), 10**-6,10**-6)




"""
-------------------------------------------VEGA---------------------------------------------------------------------
"""
if args.Vega == True:
    analisi_stella("Vega",T_vega,10**-8,5*10**-6)

if args.Vega2 == True:
    analisi_differente_stella("Vega",T_vega,10**-8,5*10**-6, 2*(10**-7), 10**-6, 4*(10**-7))

    



"""
-------------------------------------------RIGEL---------------------------------------------------------------------
"""
if args.Rigel == True:
    analisi_stella("Rigel",T_rigel,10**-8,2*10**-6)
    
if args.Rigel2 == True:
    analisi_differente_stella("Rigel",T_rigel,10**-8,2*10**-6, 10**-7, 10**-6, 4*(10**-7))
