nomegiorni=['Luendì', 'Martedì', 'Mercoledì','Giovedì','Venerdì','Sabato','Domenica']
gg=nomegiorni*5
oct={}
for k in range(31):
    oct.update({k+1:gg[k+1]})
print('\n=======================================================')
print('================= Ottobre 2024 ========================')
print('=======================================================')
for k in oct:

    print('||{:5d}{:.>36}{:>12}'.format(k, oct[k], '||') )
print('=======================================================')

