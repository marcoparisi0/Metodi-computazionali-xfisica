import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a1=np.arange(1,31,1)
a2=([24,23,22,19,18,23,25,23,24,22,20,21,19,17,26,24,24,23,22,21,20,22,25,20,23,24,23,23,22,21])
a3=([2,2,1,2,1,3,1,2,1,2,2,3,2,1,1,2,1,1,2,2,2,1,1,2,1,1,3,1,2,2,])
a4=([0,0,0,120,120,0,0,0,0,0,55,20,133,145,0,0,0,0,0,0,56,0,0,0,0,0,0,0,100,88])
a5=([0,0,0,10,20,0,0,0,0,0,4,3,15,13,0,0,0,0,0,0,2,0,0,0,0,0,0,0,12,7])
days = pd.DataFrame(columns= ['giorno','temperatura', 'errore_T','mm_pioggia','errore_mm'])
days['giorno']=a1
days['temperatura']=a2
days['errore_T']=a3
days['mm_pioggia']=a4
days['errore_mm']=a5
days.to_csv('days.csv', index=False)
