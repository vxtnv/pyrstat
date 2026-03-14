
"""
pyrstat.multivariate_timeseries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multivariate Time Series Wrapper:
  - VAR / VARselect    (vars)
  - serial.test        (vars)
  - causality          (vars)
  - irf                (vars)
  - ca.jo / vec2var    (vars + urca)
  - ur.df              (urca)

Uses rpackagemanager for automatic R package provisioning.
"""

from typing import Optional, List
import subprocess

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from rpackagemanager import rpackagemanager

# ── R-Pakete einmalig installieren & laden ──────────────────────────────

_rpkg = rpackagemanager()
for _pkg in ["vars", "urca"]:
    _rpkg.install_packages(_pkg)

_r_vars = importr("vars")
from pyrstat.univariate_timeseries import _unwrap, RTimeSeries
_r_urca = importr("urca")
_r_stats = importr("stats")
_r_base = importr("base")





# ── VARselect ───────────────────────────────────────────────────────────

def VARselect(y, lag_max: int = 10, type: str = "const", exogen=None, pretty: bool = True):
    import pandas as pd

    # pd.Series oder pd.DataFrame automatisch konvertieren
    if isinstance(y, pd.Series):
        y = ro.FloatVector(y.values)
    elif isinstance(y, pd.DataFrame):
        y = ro.FloatVector(y.values.flatten())

    kwargs = {"lag.max": lag_max, "type": type}
    if exogen is not None:
        kwargs["exogen"] = exogen

    y = _unwrap(y)
    if exogen is not None:
        exogen = _unwrap(exogen)
    result = _r_vars.VARselect(y, **kwargs)

    if pretty:
        captured = ro.r("capture.output")(result)
        print("\n".join(list(captured)))

    return result


# ── VAR ─────────────────────────────────────────────────────────────────

def VAR(y, p: int = 1, type: str = "const", exogen=None, pretty: bool = False):
    """
    VAR-Modell schätzen – wie R's VAR().

    Parameters
    ----------
    y : R ts/mts-Objekt
    p : int
        Lag-Ordnung (default 1).
    type : str
        "const" (default), "trend", "both", oder "none".
    exogen : R-Objekt | None
        Exogene Variablen (optional).
    pretty : bool

    Returns
    -------
    R varest-Objekt.
    """
    kwargs = {"p": p, "type": type}
    if exogen is not None:
        kwargs["exogen"] = exogen

    y = _unwrap(y)
    if exogen is not None:
        exogen = _unwrap(exogen)
    model = _r_vars.VAR(y, **kwargs)

    if pretty:
        summary_var(model)

    return model

# ── summary_var ─────────────────────────────────────────────────────────

def summary_var(model):
    captured = ro.r("capture.output")(ro.r("summary")(model))
    lines = list(captured)

    filtered = []
    skip_call = False
    for line in lines:
        if line.strip().startswith("Call:"):
            filtered.append("Call:")
            filtered.append("VAR(y, p = ..., type = ...)")
            filtered.append("")
            skip_call = True
            continue
        if skip_call:
            # Call-Block endet bei der nächsten Leerzeile nach "return(result)"
            if line.strip() == "" and any(")" in prev for prev in filtered[-3:]):
                skip_call = False
            continue
        filtered.append(line)

    print("\n".join(filtered))

# ── serial.test ─────────────────────────────────────────────────────────

def serial_test(model, type: str = "PT.asymptotic", lags_pt: int = 16, pretty: bool = True):
    """
    Portmanteau Test auf Residuen-Autokorrelation – wie R's serial.test().

    Parameters
    ----------
    model : R varest-Objekt (von VAR())
    type : str
        "PT.asymptotic" (default), "PT.adjusted", "BG", oder "ES".
    lags_pt : int
        Anzahl der Lags für den Portmanteau-Test (default 16).
    pretty : bool

    Returns
    -------
    R htest-Objekt.
    """
    result = _r_vars.serial_test(model, type=type, lags_pt=lags_pt)

    if pretty:
        captured = ro.r("capture.output")(result)
        lines = list(captured)

        cleaned = []
        for line in lines:
            if line.strip().startswith("data:"):
                cleaned.append("data:  VAR residuals")
            else:
                cleaned.append(line)

        print("\n".join(cleaned))

    return result

# ── causality (Granger) ─────────────────────────────────────────────────

def causality(model, cause: str, pretty: bool = True):
    """
    Granger-Kausalitätstest – wie R's causality().

    Parameters
    ----------
    model : R varest-Objekt (von VAR())
    cause : str
        Name der verursachenden Variable, z.B. "deu".
    pretty : bool

    Returns
    -------
    R causality-Objekt (enthält $Granger und $Instant).
    """
    result = _r_vars.causality(model, cause=cause)

    if pretty:
        # Granger-Test ausgeben
        granger = result.rx2("Granger")
        captured = ro.r("capture.output")(granger)
        lines = list(captured)

        cleaned = []
        for line in lines:
            if line.strip().startswith("data:"):
                cleaned.append("data:  VAR model")
            else:
                cleaned.append(line)

        print("\n".join(cleaned))

    return result

# ── irf ─────────────────────────────────────────────────────────────────

def irf(model, impulse: Optional[str] = None, response: Optional[str] = None,
        n_ahead: int = 10, ortho: bool = True, runs: int = 100,
        plot: bool = True, plot_path: str = "/tmp/pyrstat_irf.pdf"):
    """
    Impulse Response Functions – wie R's irf().

    Parameters
    ----------
    model : R varest- oder vec2var-Objekt
    impulse : str | None
        Name der Impulsvariable (None = alle).
    response : str | None
        Name der Responsevariable (None = alle).
    n_ahead : int
        Horizont (default 10).
    ortho : bool
        Orthogonalisierte IRFs (default True).
    runs : int
        Bootstrap-Runs für Konfidenzintervalle (default 100).
    plot : bool
        Plot als PDF öffnen (default True).
    plot_path : str

    Returns
    -------
    R irf-Objekt.
    """
    kwargs = {"n.ahead": n_ahead, "ortho": ortho, "runs": runs}

    if impulse is not None:
        kwargs["impulse"] = impulse
    if response is not None:
        kwargs["response"] = response

    result = _r_vars.irf(model, **kwargs)

    if plot:
        ro.globalenv["pyrstatirfresult"] = result
        ro.r('pdf("{}")'.format(plot_path))
        ro.r("plot(pyrstatirfresult)")
        ro.r("dev.off()")
        subprocess.Popen(["open", plot_path])

    return result

# ── ca.jo (Johansen-Kointegration) ─────────────────────────────────────

def ca_jo(y, K: int = 2, ecdet: str = "const", type: str = "trace",
          spec: str = "longrun", pretty: bool = True):
    """
    Johansen-Kointegrationstest – wie R's ca.jo() aus urca.

    Parameters
    ----------
    y : R ts/mts-Objekt
    K : int
        Lag-Ordnung in Levels (default 2).
    ecdet : str
        "const" (default), "trend", oder "none".
    type : str
        "trace" (default) oder "eigen".
    spec : str
        "longrun" (default) oder "transitory".
    pretty : bool

    Returns
    -------
    R ca.jo-Objekt.
    """
    y = _unwrap(y)
    result = _r_urca.ca_jo(y, K=K, ecdet=ecdet, type=type, spec=spec)

    if pretty:
        captured = ro.r("capture.output")(ro.r("summary")(result))
        print("\n".join(list(captured)))

    return result

# ── ur.df (ADF via urca) ────────────────────────────────────────────────

def ur_df(y, type: str = "drift", lags: int = 1, pretty: bool = True):
    """
    Augmented Dickey-Fuller Test via urca – wie R's ur.df().

    Parameters
    ----------
    y : R ts-Objekt oder Vektor
    type : str
        "none", "drift" (default), oder "trend".
    lags : int
        Anzahl der Lags (default 1).
    pretty : bool

    Returns
    -------
    R ur.df-Objekt.
    """
    y = _unwrap(y)
    result = _r_urca.ur_df(y, type=type, lags=lags)

    if pretty:
        captured = ro.r("capture.output")(ro.r("summary")(result))
        print("\n".join(list(captured)))

    return result

# ── vec2var ─────────────────────────────────────────────────────────────

def vec2var(cajobj, r: int = 1):
    """
    VECM → VAR-Objekt konvertieren – wie R's vec2var().

    Parameters
    ----------
    cajobj : R ca.jo-Objekt (von ca_jo())
    r : int
        Kointegrationsrang (default 1).

    Returns
    -------
    R vec2var-Objekt (kann an irf(), summary_var() etc. übergeben werden).
    """
    return _r_vars.vec2var(cajobj, r=r)

# ── Acf für multivariate Residuen ───────────────────────────────────────

def Acf(y, plot: bool = True, plot_path: str = "/tmp/pyrstat_acf.pdf", pretty: bool = True):
    """
    ACF (auch Cross-Correlation) – wie R's forecast::Acf().

    Parameters
    ----------
    y : R ts/mts-Objekt oder cbind() von Vektoren
    plot : bool
        Plot als PDF öffnen (default True).
    plot_path : str
    pretty : bool
    """
    from pyrstat.univariate_timeseries import _r_forecast
    y = _unwrap(y)

    if plot:
        ro.globalenv["pyrstatacfdata"] = y
        ro.r('pdf("{}")'.format(plot_path))
        ro.r("forecast::Acf(pyrstatacfdata)")
        ro.r("dev.off()")
        subprocess.Popen(["open", plot_path])
    else:
        result = _r_forecast.Acf(y, plot=False)
        if pretty:
            captured = ro.r("capture.output")(result)
            print("\n".join(list(captured)))
        return result


def cbind(*args, **kwargs):
    """cbind – wie R's cbind(). Unterstuetzt benannte Argumente fuer Spaltennamen."""
    unwrapped = [_unwrap(a) for a in args]
    named = {k: _unwrap(v) for k, v in kwargs.items()}
    all_args = unwrapped + list(named.values())
    result = ro.r("cbind")(*all_args)
    if named:
        n = len(all_args)
        ro.globalenv["pyrstatcbindtmp"] = result
        ro.r('pyrstatcbindtmp <- matrix(pyrstatcbindtmp, ncol={})'.format(n))
        col_str = ', '.join(['"' + c + '"' for c in named.keys()])
        ro.r('colnames(pyrstatcbindtmp) <- c({})'.format(col_str))
        result = ro.globalenv["pyrstatcbindtmp"]
    return result