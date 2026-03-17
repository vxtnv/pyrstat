"""
pyrstat.core
~~~~~~~~~~~~
Core wrappers around R econometrics functions:
  - lm / glm / summary (stats)
  - vcovHC       (sandwich)
  - coeftest     (lmtest)
  - linearHypothesis (car)

Uses rpackagemanager for automatic R package provisioning.
"""

from typing import Optional, Union, List

import numpy as np
import pandas as pd

from rpackagemanager import rpackagemanager

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri, Formula
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter


# ── R-Pakete einmalig installieren & laden ──────────────────────────────

_rpkg = rpackagemanager()
for _pkg in ["sandwich", "lmtest", "car", "ivreg"]:
    _rpkg.install_packages(_pkg)

_r_ivreg = importr("ivreg")
_r_utils = importr("utils")
_r_sandwich = importr("sandwich")
_r_lmtest   = importr("lmtest")
_r_car      = importr("car")
_r_stats    = importr("stats")
_r_base     = importr("base")
_r_print    = ro.r("print")


# ── Helpers ─────────────────────────────────────────────────────────────

def _df_to_r(df: pd.DataFrame) -> ro.DataFrame:
    """Konvertiert ein pandas DataFrame → R data.frame."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def _r_matrix_to_df(robj, row_names=None, col_names=None) -> pd.DataFrame:
    """Konvertiert eine R-Matrix → pandas DataFrame."""
    with localconverter(ro.default_converter + numpy2ri.converter):
        mat = np.array(robj)
    if row_names is None:
        try:
            row_names = list(ro.r("rownames")(robj))
        except Exception:
            row_names = None
    if col_names is None:
        try:
            col_names = list(ro.r("colnames")(robj))
        except Exception:
            col_names = None
    return pd.DataFrame(mat, index=row_names, columns=col_names)


def _resolve_model(model):
    """Gibt ein R lm-Objekt und den Formel-String zurück."""
    if isinstance(model, tuple):
        formula_str, data = model
        r_df = _df_to_r(data)
        ro.globalenv["_pyrstat_df_"] = r_df
        r_model = _r_stats.lm(Formula(formula_str), data=ro.globalenv["_pyrstat_df_"])
        return r_model, formula_str
    return model, "..."


# ── coef / fitted.values / residuals ─────────────────────────────────────
def coef(model) -> pd.Series:
    """
    Koeffizienten eines lm-Modells extrahieren – wie R's coef().
    Parameters
    ----------
    model : R lm-Objekt (von pyrstat.lm())
    Returns
    -------
    pd.Series mit den geschätzten Koeffizienten.
    """
    r_coef = _r_stats.coef(model)
    names = list(ro.r("names")(r_coef))
    values = list(r_coef)
    return pd.Series(values, index=names, name="coef")
def fitted_values(model) -> pd.Series:
    """
    Fitted Values eines lm-Modells extrahieren – wie R's fitted.values().
    Parameters
    ----------
    model : R lm-Objekt (von pyrstat.lm())
    Returns
    -------
    pd.Series mit den gefitteten Werten.
    """
    r_fitted = _r_stats.fitted_values(model)
    with localconverter(ro.default_converter + numpy2ri.converter):
        arr = np.array(r_fitted)
    return pd.Series(arr.flatten(), name="fitted_values")

def residuals(model, equation: str = None):
    """
    Residuen extrahieren – wie R's residuals().
    Gibt ein R-Objekt zurück (wie in R), das direkt an andere R-Funktionen übergeben werden kann.
    """
    r_class = list(ro.r("class")(model))

    # VAR-Modell
    if "varest" in r_class:
        varresult = model.rx2("varresult")
        if equation is not None:
            return varresult.rx2(equation).rx2("residuals")
        names = list(ro.r("names")(varresult))
        return {name: varresult.rx2(name).rx2("residuals") for name in names}

    # Alle anderen (lm, Arima, ivreg, ...)
    return _r_stats.residuals(model)

# ── lm ──────────────────────────────────────────────────────────────────

def lm(formula: str, data):
    """
    Lineare Regression fitten – genau wie R's lm().

    Parameters
    ----------
    formula : str
        R-style Formel, z.B. "y ~ x1 + x2".
    data : pd.DataFrame
        Pandas DataFrame mit den Daten.

    Returns
    -------
    R lm-Objekt (kann an summary, coeftest, vcovHC, linearHypothesis übergeben werden).
    """
    if isinstance(data, pd.DataFrame):
        r_df = _df_to_r(data)
    else:
        from pyrstat.univariate_timeseries import _unwrap
        r_df = _unwrap(data)
    ro.globalenv["pyrstatdf"] = r_df
    model = _r_stats.lm(Formula(formula), data=ro.globalenv["pyrstatdf"])
    # Formel-String am Objekt speichern für summary()
    model._pyrstat_formula = formula
    return model


# ── glm ──────────────────────────────────────────────────────────────────

def glm(
    formula: str,
    data,
    family: str = "gaussian",
):
    """
    Generalisiertes Lineares Modell fitten – wie R's glm().

    Parameters
    ----------
    formula : str
        R-style Formel, z.B. "y ~ x1 + x2".
    data : pd.DataFrame
        Pandas DataFrame mit den Daten.
    family : str
        Verteilungsfamilie als String, z.B.:
        - "gaussian"            (default, = lm)
        - "binomial"            (Logit)
        - "binomial(link='probit')"
        - "poisson"
        - "Gamma"
        - "inverse.gaussian"
        - "quasi"
        Wird intern via ro.r() evaluiert, d.h. R-Syntax ist erlaubt.

    Returns
    -------
    R glm-Objekt (kann an summary, coeftest, vcovHC, linearHypothesis
    übergeben werden).
    """
    if isinstance(data, pd.DataFrame):
        r_df = _df_to_r(data)
    else:
        from pyrstat.univariate_timeseries import _unwrap
        r_df = _unwrap(data)

    ro.globalenv["_pyrstat_df_"] = r_df

    # family als R-Objekt evaluieren (erlaubt z.B. "binomial(link='probit')")
    r_family = ro.r(family)

    model = _r_stats.glm(
        Formula(formula),
        data=ro.globalenv["_pyrstat_df_"],
        family=r_family,
    )
    model._pyrstat_formula = formula
    return model







def head(y, n: int = 6):
    """Erste n Elemente – wie R's head(). Negative n = alles außer die letzten |n|."""
    from pyrstat.univariate_timeseries import _unwrap, RTimeSeries
    y = _unwrap(y)
    return RTimeSeries(_r_utils.head(y, n))
# ── summary ─────────────────────────────────────────────────────────────

def summary(model):
    """
    R-style summary() – generische Funktion wie in R.
    Funktioniert für lm, glm, VAR, Arima, ivreg, etc.
    """
    r_class = list(ro.r("class")(model))

    captured = ro.r("capture.output")(ro.r("summary")(model))
    lines = list(captured)

    # Call-Block bereinigen (ABI-Modus dumpt den ganzen Quellcode)
    filtered = []
    skip = False
    for line in lines:
        if line.strip().startswith("Call:"):
            filtered.append("Call:")
            formula_str = getattr(model, "_pyrstat_formula", None)
            if formula_str:
                if "glm" in r_class:
                    prefix = "glm"
                elif "ivreg" in r_class:
                    prefix = "ivreg"
                else:
                    prefix = "lm"
                filtered.append(prefix + "(formula = " + formula_str + ")")
            filtered.append("")
            skip = True
            continue
        if skip:
            # Call-Block endet bei nächstem bekannten Abschnitt
            if any(line.strip().startswith(s) for s in [
                "Residual", "Estimation", "Coefficients", "---",
                "Covariance", "Correlation", "Series:", "ARIMA",
                "Deviance", "AIC", "Number", "Dispersion",
            ]):
                skip = False
            else:
                continue
        if not skip:
            filtered.append(line)

    print("\n".join(filtered))




# ── vcovHC ──────────────────────────────────────────────────────────────

def vcovHC(
    model,
    type: str = "HC1",
    omega: Optional[str] = None,
    sandwich: bool = True,
    pretty: bool = True,
) -> pd.DataFrame:
    """
    Heteroskedasticity-Consistent (HC) Kovarianzmatrix-Schätzer.

    Parameters
    ----------
    model : R lm-Objekt (von pyrstat.lm())
    type : str
        HC-Variante: "HC0", "HC1" (default), "HC2", "HC3", "HC4", "HC4m", "HC5".
    omega : str | None
        Optionale custom omega function (als R-String).
    sandwich : bool
        Ob die volle Sandwich-Form verwendet wird (default True).
    pretty : bool
        R-style Output in der Konsole (default True).

    Returns
    -------
    pd.DataFrame
        Die geschätzte Kovarianzmatrix.
    """
    kwargs = {"type": type, "sandwich": sandwich}
    if omega is not None:
        kwargs["omega"] = ro.r(omega)

    vcov = _r_sandwich.vcovHC(model, **kwargs)

    if pretty:
        _r_print(vcov)

    return _r_matrix_to_df(vcov)


# ── coeftest ────────────────────────────────────────────────────────────

def coeftest(
    model,
    vcov_func=None,
    vcov_type: str = "HC1",
    df: Optional[int] = None,
    pretty: bool = True,
) -> pd.DataFrame:
    """
    Koeffizienten-Signifikanztest (mit optionalem robusten SE).

    Parameters
    ----------
    model : R lm-Objekt (von pyrstat.lm())
    vcov_func : callable | str | None
        - None   → Standard-SE (keine Korrektur)
        - "HC0"–"HC5" Shortcut → nutzt vcovHC mit diesem Typ
        - callable  → wird direkt übergeben
    vcov_type : str
        Wird nur genutzt, wenn vcov_func als String übergeben wird.
    df : int | None
        Freiheitsgrade (optional override).
    pretty : bool
        R-style Output in der Konsole (default True).

    Returns
    -------
    pd.DataFrame
        Tabelle mit Estimate, Std. Error, t/z value, Pr(>|t/z|).
    """
    kwargs = {}

    if vcov_func is not None:
        if isinstance(vcov_func, str):
            vcov_matrix = _r_sandwich.vcovHC(model, type=vcov_func)
            kwargs["vcov."] = vcov_matrix
        else:
            kwargs["vcov."] = vcov_func
    elif vcov_type:
        vcov_matrix = _r_sandwich.vcovHC(model, type=vcov_type)
        kwargs["vcov."] = vcov_matrix

    if df is not None:
        kwargs["df"] = df

    result = _r_lmtest.coeftest(model, **kwargs)

    if pretty:
        _r_print(result)

    return _r_matrix_to_df(result)


# ── linearHypothesis ────────────────────────────────────────────────────

def linearHypothesis(
    model,
    hypothesis: Union[str, List[str]],
    vcov_func=None,
    vcov_type: Optional[str] = None,
    white_adjust: Union[str, bool] = False,
    verbose: bool = False,
    pretty: bool = True,
) -> pd.DataFrame:
    """
    Lineare Hypothesentests (Wald-Tests).

    Parameters
    ----------
    model : R lm-Objekt (von pyrstat.lm())
    hypothesis : str oder list[str]
        z.B. "x1 = 0" oder ["x1 = 0", "x2 = x3"].
    vcov_func : callable | str | None
        Robuste Kovarianzmatrix (wie bei coeftest).
    vcov_type : str
        HC-Variante falls vcov_func ein String ist.
    white_adjust : str | bool
        Shortcut für White-korrigierte Tests:
        - "hc0"–"hc4" oder True (= "hc3").
    verbose : bool
        Gibt die vollständige R-Ausgabe in der Konsole aus.
    pretty : bool
        R-style Output in der Konsole (default True).

    Returns
    -------
    pd.DataFrame
        ANOVA-ähnliche Tabelle mit Res.Df, RSS/Df, F/Chisq, Pr(>F) etc.
    """
    # Hypothesen → R character vector
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    r_hyp = ro.StrVector(hypothesis)

    kwargs = {"verbose": verbose}

    # Robuste Varianz
    if white_adjust:
        if isinstance(white_adjust, bool):
            kwargs["white.adjust"] = "hc3"
        else:
            kwargs["white.adjust"] = white_adjust
    elif vcov_func is not None:
        if isinstance(vcov_func, str):
            vcov_matrix = _r_sandwich.vcovHC(model, type=vcov_func)
            kwargs["vcov."] = vcov_matrix
        else:
            kwargs["vcov."] = vcov_func
    elif vcov_type:
        vcov_matrix = _r_sandwich.vcovHC(model, type=vcov_type)
        kwargs["vcov."] = vcov_matrix

    result = _r_car.linearHypothesis(model, r_hyp, **kwargs)

    if pretty:
        _r_print(result)

    # Ergebnis → DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_result = ro.conversion.rpy2py(result)

    if not isinstance(df_result, pd.DataFrame):
        df_result = _r_matrix_to_df(result)

    return df_result


# ── bptest ──────────────────────────────────────────────────────────────

def bptest(
    model,
    varformula: Optional[str] = None,
    studentize: bool = True,
    pretty: bool = True,
):
    kwargs = {"studentize": studentize}

    if varformula is not None:
        kwargs["varformula"] = Formula(varformula)

    result = _r_lmtest.bptest(model, **kwargs)

    if pretty:
        formula_str = getattr(model, "_pyrstat_formula", "...")
        captured = ro.r("capture.output")(result)
        lines = list(captured)

        # data:-Zeile bereinigen (ABI-Modus dumpt structure(list(...)))
        cleaned = []
        for line in lines:
            if line.strip().startswith("data:"):
                cleaned.append("data:  " + formula_str)
            else:
                cleaned.append(line)

        print("\n".join(cleaned))


def reset(
    model,
    power: int = 3,
    type: str = "fitted",
    pretty: bool = True,
):
    """
    Ramsey's RESET Test auf Fehlspezifikation.

    Parameters
    ----------
    model : R lm-Objekt (von pyrstat.lm())
    power : int
        Maximale Potenz der fitted values (default 3 → testet 2 und 3).
    type : str
        "fitted" (default), "regressor", oder "princomp".
    pretty : bool
        R-style Output in der Konsole (default True).
    """
    kwargs = {"type": type}

    # power als Sequenz 2:power übergeben, wie R es intern erwartet
    kwargs["power"] = ro.IntVector(list(range(2, power + 1)))

    result = _r_lmtest.resettest(model, **kwargs)

    if pretty:
        formula_str = getattr(model, "_pyrstat_formula", "...")
        captured = ro.r("capture.output")(result)
        lines = list(captured)

        cleaned = []
        for line in lines:
            if line.strip().startswith("data:"):
                cleaned.append("data:  " + formula_str)
            else:
                cleaned.append(line)

        print("\n".join(cleaned))




def ivreg(formula: str, data: pd.DataFrame):
    """
    Instrumentalvariablen-Regression – wie R's ivreg().

    Parameters
    ----------
    formula : str
        R-style Formel mit | für Instrumente,
        z.B. "log(y) ~ log(x1) + log(x2) | log(x2) + log(z1) + log(z2)"
    data : pd.DataFrame
        Pandas DataFrame mit den Daten.

    Returns
    -------
    R ivreg-Objekt.
    """
    r_df = _df_to_r(data)
    ro.globalenv["_pyrstat_df_"] = r_df
    model = _r_ivreg.ivreg(Formula(formula), data=ro.globalenv["_pyrstat_df_"])
    model._pyrstat_formula = formula
    return model

def summary_ivreg(model):
    """
    R-style summary() für ivreg-Modelle.

    Parameters
    ----------
    model : R ivreg-Objekt (von pyrstat.ivreg())
    """
    formula_str = getattr(model, "_pyrstat_formula", "...")

    result = _r_base.summary(model)

    captured = ro.r("capture.output")(result)
    lines = list(captured)

    filtered = []
    skip = False
    for line in lines:
        if line.strip().startswith("Call:"):
            skip = True
            filtered.append("Call:")
            filtered.append("ivreg(formula = " + formula_str + ")")
            filtered.append("")
            continue
        if skip:
            if line.strip() == "":
                skip = False
            continue
        filtered.append(line)

    print("\n".join(filtered))


