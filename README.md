# pyrstat

R econometrics in Python — powered by rpy2 and rpackagemanager.

pyrstat brings the power of R's econometric ecosystem to Python, providing well-known statistical functions from packages like **sandwich**, **lmtest**, **car**, **ivreg**, **forecast**, **tseries**, **aTSA**, **vars**, **urca**, and **zoo**.

## Requirements

- Python 3.11+
- R (installed locally)
- rpy2 (ABI mode)

## Installation

```bash
pip install --force-reinstall "git+https://github.com/vxtnv/pyrstat.git"
```

All R dependencies are installed automatically via [rpackagemanager](https://github.com/vxtnv/RPackageManager).

## Quick Start

```python
import pandas as pd
from pyrstat import lm, ivreg, summary_ivreg

data = pd.read_csv("demand.csv")

# OLS regression
model = lm("log(q) ~ log(p) + income", data)

# Instrumental variables regression
iv_model = ivreg("log(q) ~ log(p) + income | income + z", data)
summary(iv_model)
```

## Core API

### Linear and Generalized Linear Models

```python
from pyrstat import lm, glm, summary, coef, fitted_values, residuals
from pyrstat import vcovHC, coeftest, bptest, reset, linearHypothesis

model = lm("y ~ x1 + x2", data)
summary(model)                      # print regression summary
coef(model)                         # extract coefficients
fitted_values(model)               # fitted values
residuals(model)                    # residuals
vcovHC(model)                      # heteroskedasticity-robust covariance
coeftest(model)                    # coefficient significance tests
bptest(model)                      # Breusch-Pagan heteroskedasticity test
reset(model)                       # RESET specification test
linearHypothesis(model, "x1 = x2") # linear restriction test
```

### Classification

```python
from pyrstat import confusionMatrix, mlogit, postResample

# Multinomial logit
result = mlogit("y ~ x1 + x2", data)
confusionMatrix(result)
postResample(result)
```

### Time Series (Univariate)

```python
from pyrstat.univariate_timeseries import (
    ts, Arima, auto_arima, summary_arima,
    tsdisplay, Box_test, monthplot, diff, tsplot,
    kpss_test, adf_test, pp_test, checkresiduals,
    forecast, ndiffs, nsdiffs,
)

ts_data = ts(my_dataframe, date_col="date", value_col="value")
arima_model = auto_arima(ts_data)
summary_arima(arima_model)
forecast(arima_model, h=12)
kpss_test(ts_data)    # KPSS stationarity test
adf_test(ts_data)      # Augmented Dickey-Fuller test
```

### Time Series (Multivariate)

```python
from pyrstat.multivariate_timeseries import (
    VAR, VARselect, summary_var,
    serial_test, causality, irf,
    ca_jo, ur_df, vec2var, Acf, cbind,
)

var_model = VAR(ts_dataframe)
VARselect(var_model)
summary_var(var_model)
serial_test(var_model)   # serial correlation test
causality(var_model)      # Granger causality
irf(var_model)            # impulse response function
```

## R Packages Used

sandwich · lmtest · car · ivreg · forecast · tseries · aTSA · vars · urca · zoo

All R packages are managed by [rpackagemanager](https://github.com/vxtnv/RPackageManager).
