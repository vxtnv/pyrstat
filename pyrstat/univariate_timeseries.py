
"""
pyrstat.univariate_timeseries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Time Series Wrapper:
  - ts          (stats)
  - Arima       (forecast)
  - auto_arima  (forecast)
  - summary für Arima-Modelle

Uses rpackagemanager for automatic R package provisioning.
"""

from typing import Optional, List

import numpy as np
import pandas as pd

from rpackagemanager import rpackagemanager

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri, Formula
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# ── R-Pakete einmalig installieren & laden ──────────────────────────────

_rpkg = rpackagemanager()
for _pkg in ["forecast", "tseries", "aTSA","zoo", "ggplot2"]:
    _rpkg.install_packages(_pkg)



_r_aTSA = importr("aTSA")
_r_forecast = importr("forecast")
_r_tseries  = importr("tseries")
_r_stats    = importr("stats")
_r_base     = importr("base")
# ── RTimeSeries Wrapper ─────────────────────────────────────────────────
class RTimeSeries:
    """Wrapper um R ts/mts-Objekte mit R-aehnlicher Syntax."""
    def __init__(self, r_obj):
        self._r = r_obj
    def __getitem__(self, key):
        if isinstance(key, str):
            return RTimeSeries(self._r.rx(True, key))
        elif isinstance(key, (list, tuple)):
            return RTimeSeries(self._r.rx(True, ro.StrVector(key)))
        return self._r[key]
    def __repr__(self):
        return "\n".join(list(ro.r("capture.output")(self._r)))
def _unwrap(y):
    """Extrahiert das R-Objekt aus RTimeSeries, falls noetig."""
    if isinstance(y, RTimeSeries):
        return y._r
    return y


# ── ts ──────────────────────────────────────────────────────────────────

def ts(data, start=None, frequency=1):
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            r_data = ro.FloatVector(data.iloc[:, 0].values)
        else:
            # Multivariate ts (mts)
            matrix_data = ro.r("matrix")(
                ro.FloatVector(data.values.flatten(order='F')),
                nrow=data.shape[0],
                ncol=data.shape[1]
            )
            ro.globalenv["pyrstatmatrix"] = matrix_data
            col_str = ', '.join(['"' + c + '"' for c in data.columns])
            ro.r('colnames(pyrstatmatrix) <- c({})'.format(col_str))
            r_data = ro.globalenv["pyrstatmatrix"]
    elif isinstance(data, pd.Series):
        r_data = ro.FloatVector(data.values)
    else:
        r_data = ro.FloatVector(np.array(data).flatten())

    kwargs = {"frequency": frequency}
    if start is not None:
        if isinstance(start, (list, tuple)):
            kwargs["start"] = ro.IntVector(start)
        else:
            kwargs["start"] = start

    return RTimeSeries(_r_stats.ts(r_data, **kwargs))

def lag(y, k: int = 1):
    """
    Lag einer Zeitreihe – wie R's stats::lag().
    
    Parameters
    ----------
    y : R ts-Objekt
    k : int
        Lag-Ordnung. Negativ = Verzögerung (default in ECM-Kontext: -1).
    """
    y = _unwrap(y)
    return RTimeSeries(_r_stats.lag(y, k))






# ── Arima ───────────────────────────────────────────────────────────────

def Arima(
    y,
    order: List[int] = [0, 0, 0],
    seasonal: List[int] = [0, 0, 0],
    include_mean: bool = True,
    include_drift: bool = False,
    method: str = "CSS-ML",
):
    """
    ARIMA-Modell fitten – wie R's Arima() aus forecast.

    Parameters
    ----------
    y : R ts-Objekt (von pyrstat.ts())
    order : list[int]
        (p, d, q) – z.B. [2, 1, 0].
    seasonal : list[int]
        (P, D, Q) – z.B. [0, 1, 1].
    include_mean : bool
        Konstante einschließen (default True).
    include_drift : bool
        Drift-Term (default False).
    method : str
        "CSS-ML" (default), "ML", oder "CSS".

    Returns
    -------
    R Arima-Objekt.
    """
    kwargs = {
        "order": ro.IntVector(order),
        "seasonal": ro.IntVector(seasonal),
        "include.mean": include_mean,
        "include.drift": include_drift,
        "method": method,
    }

    y = _unwrap(y)
    model = _r_forecast.Arima(y, **kwargs)

    # Für summary-Output speichern
    order_str = f"({order[0]},{order[1]},{order[2]})"
    seasonal_str = f"({seasonal[0]},{seasonal[1]},{seasonal[2]})"
    model._pyrstat_arima_info = f"ARIMA{order_str}{seasonal_str}"

    return model

# ── auto_arima ──────────────────────────────────────────────────────────

def auto_arima(
    y,
    seasonal: bool = True,
    stepwise: bool = True,
    approximation: bool = True,
    max_order: int = 5,
    d: Optional[int] = None,
    D: Optional[int] = None,
    pretty: bool = True,
):
    kwargs = {
        "seasonal": seasonal,
        "stepwise": stepwise,
        "approximation": approximation,
        "max.order": max_order,
    }

    if d is not None:
        kwargs["d"] = d
    if D is not None:
        kwargs["D"] = D

    y = _unwrap(y)
    model = _r_forecast.auto_arima(y, **kwargs)

    if pretty:
        captured = ro.r("capture.output")(ro.r("summary")(model))
        lines = list(captured)

        cleaned = []
        for line in lines:
            if line.strip().startswith("Series:") and "structure(" in line:
                cleaned.append("Series: ts")
            else:
                cleaned.append(line)

        print("\n".join(cleaned))

    return model

# ── summary (Arima) ─────────────────────────────────────────────────────

def summary_arima(model, name: str = "hotel"):
    """
    R-style summary() für Arima-Modelle.
    """
    captured = ro.r("capture.output")(ro.r("summary")(model))
    lines = list(captured)

    cleaned = []
    for line in lines:
        if line.strip().startswith("Series:"):
            cleaned.append("Series: " + name)
        else:
            cleaned.append(line)

    print("\n".join(cleaned))

# ── AICc / BIC Zugriff ─────────────────────────────────────────────────


def aicc(model) -> float:
    """AICc eines Arima-Modells."""
    return float(model.rx2("aicc")[0])

def bic(model) -> float:
    """BIC eines Arima-Modells."""
    return float(model.rx2("bic")[0])





import subprocess

import os
import subprocess

import subprocess

def tsdisplay(y, main: str = "", plot_path: str = "/tmp/pyrstat_tsdisplay.pdf"):
    y = _unwrap(y)
    ro.globalenv["pyrstatplots"] = y
    ro.r('pdf("{}")'.format(plot_path))
    ro.r('forecast::tsdisplay(pyrstatplots, main="{}")'.format(main))
    ro.r("dev.off()")
    subprocess.Popen(["open", plot_path])


def autoplot(y, facets: bool = True, main: str = "", plot_path: str = "/tmp/pyrstat_autoplot.pdf"):
    y = _unwrap(y)
    ro.globalenv["pyrstatplots"] = y

    ro.r('pdf("{}")'.format(plot_path))

    if facets:
        # Default: getrennte Panels (facets weglassen = R default)
        ro.r('print(zoo::autoplot.zoo(pyrstatplots, main = "{}"))'.format(main))
    else:
        # Alle Linien überlagert
        ro.r('print(zoo::autoplot.zoo(pyrstatplots, facets = NULL, main = "{}"))'.format(main))

    ro.r("dev.off()")
    subprocess.Popen(["open", plot_path])



def monthplot(y, main: str = "", plot_path: str = "/tmp/pyrstat_monthplot.pdf"):
    """Monthplot – wie R's monthplot()."""
    y = _unwrap(y)
    ro.globalenv["pyrstatplots"] = y
    ro.r('pdf("{}")'.format(plot_path))
    ro.r('monthplot(pyrstatplots, main="{}")'.format(main))
    ro.r("dev.off()")
    subprocess.Popen(["open", plot_path])

def diff(y, lag: int = 1, differences: int = 1):
    """
    Differenzierung – wie R's diff().

    Parameters
    ----------
    y : R ts-Objekt
    lag : int
        Lag der Differenz (12 = saisonale Differenz bei monatlichen Daten).
    differences : int
        Anzahl der Differenzierungen (default 1).

    Returns
    -------
    R ts-Objekt.
    """
    y = _unwrap(y)
    return RTimeSeries(_r_base.diff(y, lag=lag, differences=differences))

def Box_test(x, lag: int = 1, fitdf: int = 0, type: str = "Ljung-Box", pretty: bool = True):
    x = _unwrap(x)
    result = _r_stats.Box_test(x, lag=lag, type=type, fitdf=fitdf)

    if pretty:
        captured = ro.r("capture.output")(result)
        lines = list(captured)

        cleaned = []
        for line in lines:
            if line.strip().startswith("data:"):
                cleaned.append("data:  residuals")
            else:
                cleaned.append(line)

        print("\n".join(cleaned))

def tsplot(y, main: str = "", plot_path: str = "/tmp/pyrstat_tsplot.pdf"):
    """ts.plot – wie R's ts.plot()."""
    y = _unwrap(y)
    ro.globalenv["pyrstatplots"] = y
    ro.r('pdf("{}")'.format(plot_path))
    ro.r('ts.plot(pyrstatplots, main="{}")'.format(main))
    ro.r("dev.off()")
    subprocess.Popen(["open", plot_path])


def kpss_test(y, null: str = "Level", pretty: bool = True):
    """
    KPSS Test – wie R's tseries::kpss.test().

    Parameters
    ----------
    y : R ts-Objekt
    null : str
        "Level" (default) oder "Trend".
    pretty : bool
    """
    y = _unwrap(y)
    result = _r_tseries.kpss_test(y, null=null)

    if pretty:
        captured = ro.r("capture.output")(result)
        lines = list(captured)

        cleaned = []
        for line in lines:
            if line.strip().startswith("data:"):
                cleaned.append("data:  ts")
            else:
                cleaned.append(line)

        print("\n".join(cleaned))

    return result


def forecast(model, h: int = 10, level: list = None, pretty: bool = True,
             plot: bool = False, plot_path: str = "/tmp/pyrstat_forecast.pdf"):
    model_r = _unwrap(model)
    ro.globalenv["_pyrstat_model_"] = model_r
    
    # ── forecast via R-String (ABI-kompatibel) ──
    cmd = f'forecast::forecast(_pyrstat_model_, h={h}'
    if level is not None:
        level_str = ', '.join(str(l) for l in level)
        cmd += f', level=c({level_str})'
    cmd += ')'
    result = ro.r(cmd)

    if pretty:
        captured = ro.r("capture.output")(result)
        print("\n".join(list(captured)))

    if plot:
        ro.globalenv["pyrstatfcresult"] = result
        ro.r('pdf("{}")'.format(plot_path))
        ro.r("plot(pyrstatfcresult)")
        ro.r("dev.off()")
        subprocess.Popen(["open", plot_path])

    return result





def ndiffs(y, alpha: float = 0.05, test: str = "kpss", max_d: int = 2):
    """
    Anzahl benoetigter Differenzierungen – wie R's forecast::ndiffs().
    Parameters
    ----------
    y : R ts-Objekt
    alpha : float
        Signifikanzniveau (default 0.05).
    test : str
        "kpss" (default), "adf", oder "pp".
    max_d : int
        Maximale Differenzierungsordnung (default 2).
    Returns
    -------
    int
    """
    y = _unwrap(y)
    result = _r_forecast.ndiffs(y, alpha=alpha, test=test, max_d=max_d)
    return int(result[0])
def nsdiffs(y, test: str = "ocsb", max_D: int = 1):
    """
    Anzahl benoetigter saisonaler Differenzierungen – wie R's forecast::nsdiffs().
    Parameters
    ----------
    y : R ts-Objekt
    test : str
        "ocsb" (default) oder "ch".
    max_D : int
        Maximale saisonale Differenzierungsordnung (default 1).
    Returns
    -------
    int
    """
    y = _unwrap(y)
    result = _r_forecast.nsdiffs(y, test=test, max_D=max_D)
    return int(result[0])
def checkresiduals(model, lag: Optional[int] = None, pretty: bool = True):
    """
    Residuen-Diagnostik + Ljung-Box Test – wie R's checkresiduals() aus forecast.
    Inkl. fitdf-Korrektur (automatisch aus dem Modell abgeleitet).

    Parameters
    ----------
    model : R Arima-Objekt (von Arima() oder auto_arima())
    lag : int | None
        Anzahl Lags für den Ljung-Box Test (None = forecast wählt automatisch).
    pretty : bool
        Ljung-Box Test Output in der Konsole (default True).
    """
    # Plot als PDF öffnen
    plot_path = "/tmp/pyrstat_checkresiduals.pdf"
    ro.r('pdf("{}")'.format(plot_path))

    kwargs = {}
    if lag is not None:
        kwargs["lag"] = lag

    # checkresiduals gibt den Ljung-Box Test als Rückgabewert
    result = _r_forecast.checkresiduals(model, **kwargs)

    ro.r("dev.off()")
    subprocess.Popen(["open", plot_path])

    if pretty:
        captured = ro.r("capture.output")(result)
        lines = list(captured)

        cleaned = []
        for line in lines:
            if line.strip().startswith("data:"):
                cleaned.append("data:  Residuals")
            else:
                cleaned.append(line)

        print("\n".join(cleaned))


def adf_test(y, column: str = None, pretty: bool = True):
    """Augmented Dickey-Fuller Test – wie R's aTSA::adf.test()."""
    y = _unwrap(y)
    if column is not None:
        y = y.rx(True, column)
    ro.globalenv["pyrstatts"] = y
    if pretty:
        captured = ro.r('capture.output(aTSA::adf.test(pyrstatts))')
        print("\n".join(list(captured)))


def pp_test(y, pretty: bool = True):
    """Phillips-Perron Test – wie R's aTSA::pp.test()."""
    y = _unwrap(y)
    ro.globalenv["pyrstatts"] = y
    if pretty:
        captured = ro.r('capture.output(aTSA::pp.test(pyrstatts))')
        print("\n".join(list(captured)))

