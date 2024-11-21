import numpy as np
import somme as s

nnn = input('Immettere un numero: ')
nn=int(nnn)
ss=s.somma(nn)
print('la somma dei primi', nn, 'numeri è', ss)
rr=s.radici(nn)
print('la somma delle radici dei  primi', nn, 'numeri è', rr)
sspp=s.sp(nn)
print('la somma e il prodotto dei primi', nn, 'numeri sono', sspp)
e = input('Immettere esponente della serie: ')
a=int(e)
paolo=s.serie(nn,a)
print('il risultato della serie è:' , paolo)
