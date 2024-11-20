import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


link= "https://raw.githubusercontent.com/s-germani/metodi-computazionali-fisica-2024/refs/heads/main/dati/moduli_scientifici/kplr010666592-2011240104155_slc.csv" #metti le virgolette
data=pd.read_csv(link)

dati= data[['TIME', 'PDCSAP_FLUX','PDCSAP_FLUX_ERR']]

fig, ax = plt.subplots(figsize=(12,6))
plt.errorbar(dati['TIME'], dati['PDCSAP_FLUX'], yerr=dati['PDCSAP_FLUX_ERR'], fmt='.', color='cornflowerblue' )
plt.xlabel('Time (BJD - 2454833)', fontsize=14)
plt.ylabel(r'Flux ($e^-/s$)',      fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#da finire, non ho capito proprio
