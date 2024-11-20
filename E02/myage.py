# import datetime
from datetime import datetime, timedelta
def eta(data_nascita):
    oggi = datetime.now()  # mi da la  data e ora attuali
    delta = oggi - data_nascita  # Differenza temporale tra oggi e la data di nascita
    anni =oggi.year-data_nascita.year
    if oggi.month<data_nascita.month or ( oggi.month == data_nascita.month and oggi.day < data_nascita.day):
        anni=oggi.year-data_nascita.year-1
    giorni = delta.days
    secondi = delta.total_seconds()

    return anni, giorni, secondi
mydate_str = input('data di nascita (formato DD-MM-YYYY):')
data_nascita = datetime.strptime(mydate_str, "%d-%m-%Y")
anni, giorni, secondi = eta(data_nascita)
print('Età in anni{:<20}'.format(anni))
print('Età in giorni{:<20}'.format(giorni))
print('Età in secondi{:<20}'.format(secondi))
