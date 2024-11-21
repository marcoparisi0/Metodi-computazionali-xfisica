import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reco import Hit

def array_m(path):
    data = pd.read_csv(path)
    hits = []     #creo una lista vuota
    for index, row in data.iterrows():     #restituisce una tupla per ogni riga del DataFrame
        hit = Hit(m=row['mod_id'], s=row['det_id'], t=row['hit_time'])
        hits.append(hit)
    return np.array(hits)


pt = input('Immettere il pathname del file csv: ')
hits = array_m(pt)

#print([str(hit) for hit in hits])  # Stampa una lista con le rappresentazioni leggibili degli oggetti Hit

# Richiedo i path dei file CSV separati da virgole
def array_tot_sort(paths):
    paths = input('Immettere i pathname dei file csv separati da virgola: ').split(',')
    for pa in paths:
        data = pd.read_csv(p.strip())  # Elimina spazi bianchi dal path e legge il file
    hits=[]
    for index, row in data.iterrows():
        hit = Hit(m=row['mod_id'], s=row['det_id'], t=row['hit_time'])
        hits.append(hit)
    hits_conc=pd.concat(hits)
    h= np.array(hits_conc)
    #  indici che ordinano la terza colonna (colonna con indice 2)
    indici_ordinati = np.argsort(h[:, 2])
    # Uso gli indici per ordinare l'intero array in base alla terza colonna
    h_sorted = h[indici_ordinati]

    return h_sorted

    
   
    



 
    
    
