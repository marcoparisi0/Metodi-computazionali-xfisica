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
    parser.add_argument('-s', '--Sun',    action='store_true', help="Mostra l'analisi completa relativa al sole")
    parser.add_argument('-sa', '--Saurigae',    action='store_true', help="Mostra l'analisi completa relativa alla stella Saurigae")
    parser.add_argument('-v', '--Vega',    action='store_true', help="Mostra l'analisi completa relativa alla stella Vega")
    parser.add_argument('-r', '--Rigel',    action='store_true', help="Mostra l'analisi completa relativa alla stella Rigel")
    return  parser.parse_args()

args = parse_arguments()



la=np.random.uniform(low=10**(-8), high=5*10**(-6), size=10000)  # lunghezze d'onda  in  m   considero spettro da UV a infrarossi


plt.hist(la, bins=25, alpha=0.8, color='green', ec='darkgreen')
plt.title("Distribuzione uniforme lunghezza d'onda")
plt.xlabel(r'$\lambda$')
plt.show()

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


"""
-------------------------------------------SOLE---------------------------------------------------------------------
"""

if args.Sun == True:
#PLOT SPETTRO  SOLE
    E=cn.B(lmbd,T_sun)
    plt.plot(lmbd,E, color='black')
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.title('Spettro di emissione del Sole, no scattering')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"radiazione emessa [$J/s m^2$]")


    E_vis = E[mask]


    for i in range(len(lmbd_vis) - 1):
        plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [E_vis[i], E_vis[i + 1]], color=colors[i])
    plt.show()





#DISTRIBUZIONE FOTONI SOLE ORIZZONTE
    Tr=cn.D_scatter(lmbd,S_o,T_sun)
    Tr_vis = Tr[mask]

    def D_scatter_min_o(l):
       return -cn.D_scatter(l,S_o,T_sun)
    ris_o=optimize.minimize(D_scatter_min_o,x0=8*(10**-7))
    l_max_o=ris_o.x[0]
    o_max=cn.D_scatter(l_max_o,S_o,T_sun)


    plt.plot(lmbd,Tr, color='black')
    plt.plot(l_max_o,o_max,'o',color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_o))
    plt.plot([l_max_o, l_max_o], [0, o_max], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering Sole ad orizzonte')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")

    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Tr_vis[i], Tr_vis[i + 1]], color=colors[i])
    plt.show()






#SOLE ZENITH
    Z=cn.D_scatter(lmbd,S_z,T_sun)
    Z_vis = Z[mask]

    def D_scatter_min_z(l):
       return -cn.D_scatter(l,S_z,T_sun)
    ris_z=optimize.minimize(D_scatter_min_z,x0=5*(10**-7))
    l_max_z=ris_z.x[0]
    z_max=cn.D_scatter(l_max_z,S_z,T_sun)

    plt.plot(lmbd,Z, color='black')
    plt.plot(l_max_z,z_max,'o', color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_z))
    plt.plot([l_max_z, l_max_z], [0, z_max], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering Sole allo zenith')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")
    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Z_vis[i], Z_vis[i + 1]], color=colors[i])

    plt.show()



#confronto
    plt.plot(lmbd,cn.D(lmbd,T_sun), color='goldenrod',label='senza assorbimento')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_z,T_sun),  color='gold',label='scattering posizione Zenith')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_o,T_sun),  color='tan',label='scattering posizione Orizzonte')
    plt.legend()
    plt.title('Confronto distribuzioni fotoni')
    plt.xlabel(r'$\lambda$ [m]')
    plt.ylabel(r'$fotoni/sm^3$')
    plt.show()


#andamento flusso fotoni in funzione della posizione del sole
    
    f=[]
    for t in teta:
       integranda=cn.D_scatter(lmbd,cn.S_teta(t),T_sun)
       flusso=integrate.simpson(integranda,dx=0.01)
       f.append(flusso)
    plt.plot(cn.S_teta(teta),f, color='teal')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title('Andamento flusso integrato di fotoni')
    plt.xlabel('Distanza del Sole dallo Zenith [m]')
    plt.ylabel(r'$fotoni/s m^2$')
    plt.show()



"""
-------------------------------------------SAURIGAE---------------------------------------------------------------------
"""
if args.Saurigae == True:
#PLOT SPETTRO  SAURIGAE
    E1=cn.B(lmbd,T_sau)
    plt.plot(lmbd,E1, color='black')
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.title('Spettro di emissione Saurigae, no scattering')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"radiazione emessa [$J/s m^2$]")


    E_vis1 = E1[mask]
    for i in range(len(lmbd_vis) - 1):
        plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [E_vis1[i], E_vis1[i + 1]], color=colors[i])
    plt.show()




#DISTRIBUZIONE FOTONI SAURIGAE ORIZZONTE
    Tr1=cn.D_scatter(lmbd,S_o,T_sau)
    Tr_vis1 = Tr1[mask]

    def D_scatter_min_o1(l):
       return -cn.D_scatter(l,S_o,T_sau)
    ris_o1=optimize.minimize(D_scatter_min_o1,x0=10**-6)
    l_max_o1=ris_o1.x[0]
    o_max1=cn.D_scatter(l_max_o1,S_o,T_sau)


    plt.plot(lmbd,Tr1, color='black')
    plt.plot(l_max_o1,o_max1,'o',color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_o1))
    plt.plot([l_max_o1, l_max_o1], [0, o_max1], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering Saurigae ad orizzonte')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")

    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Tr_vis1[i], Tr_vis1[i + 1]], color=colors[i])
    plt.show()






#SAURIGAE ZENITH
    Z1=cn.D_scatter(lmbd,S_z,T_sau)
    Z_vis1 = Z1[mask]

    def D_scatter_min_z1(l):
       return -cn.D_scatter(l,S_z,T_sau)
    ris_z1=optimize.minimize(D_scatter_min_z1,x0=10**-6)
    l_max_z1=ris_z1.x[0]
    z_max1=cn.D_scatter(l_max_z1,S_z,T_sau)

    plt.plot(lmbd,Z1, color='black')
    plt.plot(l_max_z1,z_max1,'o', color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_z1))
    plt.plot([l_max_z1, l_max_z1], [0, z_max1], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering Saurigae allo zenith')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")
    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Z_vis1[i], Z_vis1[i + 1]], color=colors[i])

    plt.show()



#confronto
    plt.plot(lmbd,cn.D(lmbd,T_sau), color='goldenrod',label='senza assorbimento')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_z,T_sau),  color='gold',label='scattering posizione Zenith')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_o,T_sau),  color='tan',label='scattering posizione Orizzonte')
    plt.legend()
    plt.title('Confronto distribuzioni fotoni')
    plt.xlabel(r'$\lambda$ [m]')
    plt.ylabel(r'$fotoni/sm^3$')
    plt.show()


#andamento flusso fotoni in funzione della posizione di saurigae
    
    s=[]
    for t in teta:
       integranda=cn.D_scatter(lmbd,cn.S_teta(t),T_sau)
       flusso=integrate.simpson(integranda,dx=0.01)
       s.append(flusso)
    plt.plot(cn.S_teta(teta),s, color='teal')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title('Andamento flusso integrato di fotoni')
    plt.xlabel('Distanza Saurigae dallo Zenith [m]')
    plt.ylabel(r'$fotoni/s m^2$')
    plt.show()





"""
-------------------------------------------VEGA---------------------------------------------------------------------
"""
if args.Vega == True:
#PLOT SPETTRO  VEGA
    E2=cn.B(lmbd,T_vega)
    plt.plot(lmbd,E2, color='black')
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.title('Spettro di emissione Vega, no scattering')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"radiazione emessa [$J/s m^2$]")


    E_vis2 = E2[mask]
    for i in range(len(lmbd_vis) - 1):
        plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [E_vis2[i], E_vis2[i + 1]], color=colors[i])
    plt.show()




#DISTRIBUZIONE FOTONI VEGA ORIZZONTE
    Tr2=cn.D_scatter(lmbd,S_o,T_vega)
    Tr_vis2 = Tr2[mask]

    def D_scatter_min_o2(l):
       return -cn.D_scatter(l,S_o,T_vega)
    ris_o2=optimize.minimize(D_scatter_min_o2,x0=10**-6)
    l_max_o2=ris_o2.x[0]
    o_max2=cn.D_scatter(l_max_o2,S_o,T_vega)


    plt.plot(lmbd,Tr2, color='black')
    plt.plot(l_max_o2,o_max2,'o',color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_o2))
    plt.plot([l_max_o2, l_max_o2], [0, o_max2], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering Vega ad orizzonte')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")

    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Tr_vis2[i], Tr_vis2[i + 1]], color=colors[i])
    plt.show()






#VEGA  ZENITH
    Z2=cn.D_scatter(lmbd,S_z,T_vega)
    Z_vis2 = Z2[mask]

    def D_scatter_min_z2(l):
       return -cn.D_scatter(l,S_z,T_vega)
    ris_z2=optimize.minimize(D_scatter_min_z2,x0=4*(10**-7))
    l_max_z2=ris_z2.x[0]
    z_max2=cn.D_scatter(l_max_z2,S_z,T_vega)

    plt.plot(lmbd,Z2, color='black')
    plt.plot(l_max_z2,z_max2,'o', color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_z2))
    plt.plot([l_max_z2, l_max_z2], [0, z_max2], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering Vega allo zenith')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")
    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Z_vis2[i], Z_vis2[i + 1]], color=colors[i])

    plt.show()



#confronto
    plt.plot(lmbd,cn.D(lmbd,T_vega), color='goldenrod',label='senza assorbimento')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_z,T_vega),  color='gold',label='scattering posizione Zenith')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_o,T_vega),  color='tan',label='scattering posizione Orizzonte')
    plt.legend()
    plt.title('Confronto distribuzioni fotoni')
    plt.xlabel(r'$\lambda$ [m]')
    plt.ylabel(r'$fotoni/sm^3$')
    plt.show()


#andamento flusso fotoni in funzione della posizione di saurigae
    
    v=[]
    for t in teta:
       integranda=cn.D_scatter(lmbd,cn.S_teta(t),T_vega)
       flusso=integrate.simpson(integranda,dx=0.01)
       v.append(flusso)
    plt.plot(cn.S_teta(teta),v, color='teal')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title('Andamento flusso integrato di fotoni')
    plt.xlabel('Distanza Vega dallo Zenith [m]')
    plt.ylabel(r'$fotoni/s m^2$')
    plt.show()






"""
-------------------------------------------RIGEL---------------------------------------------------------------------
"""
if args.Rigel == True:
#PLOT SPETTRO  RIGEL
    E3=cn.B(lmbd,T_rigel)
    plt.plot(lmbd,E3, color='black')
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.title('Spettro di emissione Riegl, no scattering')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"radiazione emessa [$J/s m^2$]")


    E_vis3 = E3[mask]
    for i in range(len(lmbd_vis) - 1):
        plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [E_vis3[i], E_vis3[i + 1]], color=colors[i])
    plt.show()




#DISTRIBUZIONE FOTONI RIGEL ORIZZONTE
    Tr3=cn.D_scatter(lmbd,S_o,T_rigel)
    Tr_vis3 = Tr3[mask]

    def D_scatter_min_o3(l):
       return -cn.D_scatter(l,S_o,T_rigel)
    ris_o3=optimize.minimize(D_scatter_min_o3,x0=10**-6)
    l_max_o3=ris_o3.x[0]
    o_max3=cn.D_scatter(l_max_o3,S_o,T_rigel)


    plt.plot(lmbd,Tr3, color='black')
    plt.plot(l_max_o3,o_max3,'o',color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_o3))
    plt.plot([l_max_o3, l_max_o3], [0, o_max3], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering Rigel ad orizzonte')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")

    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Tr_vis3[i], Tr_vis3[i + 1]], color=colors[i])
    plt.show()






#RIGEL  ZENITH
    Z3=cn.D_scatter(lmbd,S_z,T_rigel)
    Z_vis3 = Z3[mask]

    def D_scatter_min_z3(l):
       return -cn.D_scatter(l,S_z,T_rigel)
    ris_z3=optimize.minimize(D_scatter_min_z3,x0=4*(10**-7))
    l_max_z3=ris_z3.x[0]
    z_max3=cn.D_scatter(l_max_z3,S_z,T_rigel)

    plt.plot(lmbd,Z3, color='black')
    plt.plot(l_max_z3,z_max3,'o', color='darkslateblue',label=r'$\lambda$  = {:.2e} m'.format(l_max_z3))
    plt.plot([l_max_z3, l_max_z3], [0, z_max3], linestyle='dashed', color='darkslateblue', linewidth=1)
    plt.axvline(380*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.axvline(750*(10**-9),   color='slategrey',      linewidth=0.4, linestyle='dashed')
    plt.legend()
    plt.title('Distribuzione fotoni,scattering Rigel allo zenith')
    plt.xlabel("lunghezza d'onda [m]")
    plt.ylabel(r"$fotoni/sm^3$")
    for i in range(len(lmbd_vis) - 1):
       plt.fill_between([lmbd_vis[i], lmbd_vis[i + 1]], 0, [Z_vis3[i], Z_vis3[i + 1]], color=colors[i])

    plt.show()



#confronto
    plt.plot(lmbd,cn.D(lmbd,T_rigel), color='goldenrod',label='senza assorbimento')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_z,T_rigel),  color='gold',label='scattering posizione Zenith')
    plt.plot(lmbd,cn.D_scatter(lmbd,S_o,T_rigel),  color='tan',label='scattering posizione Orizzonte')
    plt.legend()
    plt.title('Confronto distribuzioni fotoni')
    plt.xlabel(r'$\lambda$ [m]')
    plt.ylabel(r'$fotoni/sm^3$')
    plt.show()


#andamento flusso fotoni in funzione della posizione di saurigae
    
    r=[]
    for t in teta:
       integranda=cn.D_scatter(lmbd,cn.S_teta(t),T_rigel)
       flusso=integrate.simpson(integranda,dx=0.01)
       r.append(flusso)
    plt.plot(cn.S_teta(teta),r, color='teal')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title('Andamento flusso integrato di fotoni')
    plt.xlabel('Distanza Rigel dallo Zenith [m]')
    plt.ylabel(r'$fotoni/s m^2$')
    plt.show()
