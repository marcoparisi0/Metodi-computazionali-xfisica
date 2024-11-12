import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  scipy import integrate
import math
import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Esempio utilizzo argarse.')    
    parser.add_argument('-f', '--file',    action='store', help='lettura file')  #qui invece mi serve immagazinare il file da leggere, poi ci andrà il nome del file (in questo caso csv) , che leggerà e immagazzinerà  (python3 blabla  -f nomefile.csv)
    parser.add_argument('-v', '--velocità',    action='store_true',      help='grafico velocità-tempo')  #non c'è un valore da passare quindi va bene store_true, voglio solo immagazinare se è vero o falso (python3 blabla  -v)  , così mi stampera direttamente il grafico 
    parser.add_argument('-s', '--spazio',    action='store_true',  help='grafico spazio-tempo')
    return  parser.parse_args()


data=pd.read_csv('vel_vs_time.csv')
dati=data[['t','v']]
v=dati['v']
t=dati['t']


x=[]

for i in range(len(v)):
     cucu=integrate.simpson(v[:i+1], dx=t[1]-t[0])
     x.append(cucu)



     
def main():
    args = parse_arguments()

    
    if args.velocità == True:
        plt.plot(t,v, color = 'green')
        plt.xlabel('tempo')
        plt.ylabel('velocità')
        plt.show()
        
    if args.spazio == True:
        plt.plot(t,x, color = 'purple')
        plt.xlabel('tempo')
        plt.ylabel('spazio')
        plt.show()
if __name__ == '__main__':
    main()
        





"""
gli passo direttamente l'array,m fa l'integrale per ogni valore
"""
