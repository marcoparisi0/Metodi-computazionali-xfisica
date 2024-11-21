import numpy as np
def somma(n):
    s=0
    for i in range(1, n+1 ):
       s=s+i
    return s

def radici(n):
    s=0
    for i in range(1 ,n+1):
        s=s+np.sqrt(i)
    return s
def sp(n):
    """la funzione restituisce la somma e il prodotto dei numeri interi fin ad n"""
    s=0
    p=1
    for i in range(1,n+1):
        s+=i
    for i in range(1,n+1):
        p=p*i

    return s,p
def serie(n,a=1):
    """il secondo valore di input è opzionale, se non messo utilizza 1 """
    s=0
    for i in range( 1, n+1):
         s+=np.power(i,a)
    return s


    



    


    
