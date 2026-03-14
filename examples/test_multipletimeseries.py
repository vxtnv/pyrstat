
import pandas as pd
from pyrstat import (ts, diff, tsdisplay, monthplot, tsplot,
                     adf_test, VARselect, VAR, summary_var,
                     serial_test, causality, irf, lm, residuals,
                     ca_jo, ur_df, vec2var, Acf, autoplot, cbind,summary, head, lag)





# Tutorial 7 – Kointegration / VECM
yield_rates = pd.read_csv("/Users/marlons/Downloads/MW82/data/yields.csv")
yield_rates["spread"] = yield_rates["y10"] - yield_rates["m3"]
yield_rates = ts(yield_rates[["y10", "m3","spread"]], start=[1962, 1], frequency=4)


coint_eq = lm("y10 ~ m3", yield_rates)



u_spread = ts(residuals(coint_eq), start=[1962, 1], frequency=4)
u_spread_lagged = head(lag(u_spread, -1), -1)


# VECM (add lagged residual as exogenous variable), lag order can be adjusted based on residual plots
# We choose p=3 since higher lag order requires too many parameters. captured most correlation before lag=7 anyway
#and we have quarterly data
var1 = VAR(diff(yield_rates[["y10", "m3"]]), p = 3, exogen = cbind(u_spread_lagged=u_spread_lagged), type = "const")
#alpha_y10 = ~ -0.088 -> Speed of adjustment.









vec_jo = ca_jo(yield_rates[["y10", "m3"]], K=4, ecdet="const", type="trace")





var_ca = vec2var(vec_jo, r=1)
print(var_ca)
irf(var_ca, impulse="y10", ortho=True, runs=500)

