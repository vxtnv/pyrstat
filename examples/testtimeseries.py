### Testaufruf (analog zu deinem R-Skript)

import pandas as pd
from pyrstat import ts, Arima, auto_arima, summary_arima, aicc, bic,residuals
from pyrstat import lm, Arima, ts, tsdisplay, Box_test, monthplot, diff, tsplot, kpss_test, adf_test, pp_test, Box_test






hotel = pd.read_csv("/hotel.csv")

hotel_ts = ts(hotel, start=2001, frequency=12)
print(hotel_ts)



# Saisonale Differenz
hotel_sadj = diff(hotel_ts, lag=12)

# Stationaritätstests
kpss_test(hotel_sadj, null="Level")
adf_test(hotel_sadj)
pp_test(hotel_sadj)

