import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

link="https://raw.githubusercontent.com/s-germani/metodi-computazionali-fisica-2024/refs/heads/main/dati/moduli_scientifici/ExoplanetsPars_2024.csv"
data=pd.read_csv(link,comment='#')
df = data[['pl_orbper','pl_bmassj','pl_orbsmax','st_mass','discoverymethod']]
print(df)
orbital_period= data['pl_orbper']
J_mass=data['pl_bmassj']
au=data['pl_orbsmax']
solar_mass=data['st_mass']
dm=data['discoverymethod']
plt.scatter(J_mass,orbital_period, color='green') #mette i punti e basta
plt.xscale('log')  # Scala logaritmica per l'asse X
plt.yscale('log')  # Scala logaritmica per l'asse Y
plt.show()
plt.scatter(au**2/solar_mass , orbital_period, color= 'yellow' )
plt.xscale('log')  
plt.yscale('log')  
plt.show()

new_df_transito=df.loc[(data['discoverymethod'] == 'Transit')]
new_df_velocita=df.loc[(data['discoverymethod']== 'Radial Velocity')]
plt.scatter(new_df_transito['pl_bmassj'] ,new_df_transito['pl_orbper'] , color= 'green', alpha= 0.2,label='scoperta per transito')
plt.scatter(new_df_velocita['pl_bmassj'] ,new_df_velocita['pl_orbper'] , color= 'purple', alpha= 0.2,label='scoperta per velocità radiale')
plt.xlabel('massa pianeta')
plt.ylabel('periodo orbitale')
plt.legend(fontsize=14)
plt.xscale('log')  
plt.yscale('log')  
plt.show()

plt.hist ( new_df_transito['pl_bmassj'] , bins=100 ,range=(-50,50), color= 'gold', alpha= 0.2,label='scoperta per transito')
plt.hist(new_df_velocita['pl_bmassj'] ,bins= 100, range=(-50,50),  color= 'green', alpha= 0.2,label='scoperta per velocità radiale')
plt.title('istogramma massa pianeti differenziati per tipologia di scoperta')
plt.xlabel('massa pianeta misurata')
plt.ylabel('frequenza')
plt.legend(fontsize=14)
plt.show()

plt.hist ( np.log(new_df_transito['pl_bmassj']) , bins=100 ,range=(-50,50), color= 'gold', alpha= 0.5,label='scoperta per transito')
plt.hist(np.log(new_df_velocita['pl_bmassj']) ,bins= 100, range=(-50,50),  color= 'green', alpha= 0.5,label='scoperta per velocità radiale')
plt.title('istogramma massa pianeti differenziati per tipologia di scoperta')
plt.xlabel('massa pianeta misurata')
plt.ylabel('frequenza')
plt.legend(fontsize=14)
plt.show()



#ESERCIZIO 2.A
fig, axs = plt.subplots(2, 2)


axs[0,0].scatter(np.log(new_df_transito['pl_bmassj']) ,np.log(new_df_transito['pl_orbper']) , color= 'green', alpha= 0.2,label='scoperta per transito')
axs[0,0].scatter(np.log(new_df_velocita['pl_bmassj']) ,np.log(new_df_velocita['pl_orbper']) , color= 'purple', alpha= 0.2,label='scoperta per velocità radiale')
axs[0,0].set_xlabel('massa pianeta')
axs[0,0].set_ylabel('periodo orbitale')
axs[0,0].legend(fontsize=14)

axs[1,0].hist ( np.log(new_df_transito['pl_bmassj']) , bins=100 ,range=(-50,50), color= 'gold', alpha= 0.5,label='scoperta per transito')
axs[1,0].hist(np.log(new_df_velocita['pl_bmassj']) ,bins= 100, range=(-50,50),  color= 'green', alpha= 0.5,label='scoperta per velocità radiale')
axs[1,0].set_title('istogramma massa pianeti differenziati per tipologia di scoperta')
axs[1,0].set_xlabel('massa pianeta misurata')
axs[1,0].set_ylabel('frequenza')
axs[1,0].legend(fontsize=4)

axs[1,1].hist ( np.log(new_df_transito['pl_orbper']) , bins=100 ,range=(-50,50), color= 'gold', alpha= 0.5,orientation='horizontal',label='scoperta per transito')
axs[1,1].hist(np.log(new_df_velocita['pl_orbper']) ,bins= 100, range=(-50,50),  color= 'green', alpha= 0.5,orientation='horizontal', label='scoperta per velocità radiale')
axs[1,1].set_title('istogramma massa pianeti differenziati per tipologia di scoperta')
axs[1,1].set_xlabel('massa pianeta misurata')
axs[1,1].set_ylabel('frequenza')
axs[1,1].legend(fontsize=4)

# Rimuovo assi per riquadro non necessario
axs[0,1].axis('off')

plt.savefig('Exoplanets_Period_vs_Mass_Detection.pdf')
plt.savefig('Exoplanets_Period_vs_Mass_Detection.png')
plt.show()
