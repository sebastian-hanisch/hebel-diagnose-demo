"""Kern: einfache Gerade y = b0 + b1·x, Hebel, Residuen, intern und extern studentisiert, Cook-Abstand, Schwellen (numpy, vektorisiert über Datensätze).

Konvention (steht auch im 📐-Expander): p = 2 Spalten (Achsenabschnitt und Steigung), n Punkte.
  Hebel h_i = 1/n + (x_i − x̄)²/Sxx, Spur der Hat-Matrix = p, Mittel der Hebel = p/n.
  s² = SSE/(n − p); intern studentisiert t_i = e_i/(s·√(1 − h_i)); extern t*_i = t_i·√((n − p − 1)/(n − p − t_i²)).
  Cook-Abstand D_i = e_i² h_i/(p s² (1 − h_i)²) = t_i²/p · h_i/(1 − h_i) (Cook 1977, Gl. 8); bei Hebel 1 undefiniert (0/0).
Undefinierte Werte sind `nan` im Feld (nie `inf`); die Anzeige schreibt dort „undefiniert“. Hebel über 1 − 1e-9 gilt als 1."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

import hdi_constants as C

H_EINS = 1.0 - 1e-9


@dataclass(frozen=True)
class Rows:
    """Anpassung je Zeile (Datensatz): Felder der Form (R,) bzw. (R, N)."""
    n: int
    b0: np.ndarray
    b1: np.ndarray
    e: np.ndarray
    h: np.ndarray
    s2: np.ndarray
    D: np.ndarray
    t_int: np.ndarray
    t_ext: np.ndarray
    se1: np.ndarray
    se0: np.ndarray
    sxx: np.ndarray


def fit_rows(x: np.ndarray, y: np.ndarray) -> Rows:
    """Gerade je Zeile von x und y (Form (R, N)). N > p ist nötig; Zeilen mit Sxx = 0 geben nan-Felder."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n = x.shape[1]
    if n <= C.P:
        raise ValueError("Mit höchstens p = 2 Punkten ist die Streuung s² = SSE/(n − p) nicht definiert.")
    xb = x.mean(axis=1, keepdims=True)
    yb = y.mean(axis=1, keepdims=True)
    sxx = ((x - xb) ** 2).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = ((x - xb) * (y - yb)).sum(axis=1, keepdims=True) / sxx
        b0 = yb - b1 * xb
        e = y - (b0 + b1 * x)
        h = 1.0 / n + (x - xb) ** 2 / sxx
        s2 = (e ** 2).sum(axis=1, keepdims=True) / (n - C.P)
        h_eins = h > H_EINS
        hh = np.where(h_eins, np.nan, h)
        D = e ** 2 * hh / (C.P * s2 * (1.0 - hh) ** 2)
        t_int = e / np.sqrt(s2 * (1.0 - hh))
        rest = n - C.P - t_int ** 2
        t_ext = np.where(rest > 0, t_int * np.sqrt((n - C.P - 1) / np.where(rest > 0, rest, 1.0)), np.nan)
        se1 = np.sqrt(s2 / sxx)
        se0 = np.sqrt(s2 * (1.0 / n + xb ** 2 / sxx))
    return Rows(n, b0[:, 0], b1[:, 0], e, h, s2[:, 0], D, t_int, t_ext, se1[:, 0], se0[:, 0], sxx[:, 0])


def fit(x, y) -> Rows:
    """Eine Anpassung (Felder der Form (1, N))."""
    return fit_rows(np.asarray(x, dtype=float)[None, :], np.asarray(y, dtype=float)[None, :])


def schwellen(n: int) -> dict:
    """Schwellen für n Punkte: Hebel 2p/n, Cook 4/n, Cook F(p, n − p; 0,5), Cook 1, |t*| 3. Bei p = 2 sind 2p/n und 4/n dieselbe Zahl."""
    return {
        "hebel": 2.0 * C.P / n,
        "cook_4n": 4.0 / n,
        "cook_f50": float(stats.f.ppf(0.5, C.P, n - C.P)),
        "cook_1": 1.0,
        "t3": 3.0,
    }


def alarme(r: Rows, punkt: int | slice | None = None, thr: dict | None = None) -> dict:
    """Boolesche Felder, wo der Punkt (Spalte oder Teilbereich) eine Schwelle überschreitet; nan-Werte (undefiniert) zählen nicht als Alarm."""
    n = r.n
    thr = thr or schwellen(n)
    sel = slice(None) if punkt is None else punkt
    with np.errstate(invalid="ignore"):
        return {
            "hebel": r.h[:, sel] > thr["hebel"],
            "cook_4n": r.D[:, sel] > thr["cook_4n"],
            "cook_f50": r.D[:, sel] > thr["cook_f50"],
            "cook_1": r.D[:, sel] > thr["cook_1"],
            "t3": np.abs(r.t_ext[:, sel]) > thr["t3"],
        }


def max_ohne_nan(a: np.ndarray) -> np.ndarray:
    """Zeilenmaximum, nan zählt nicht (Zeile aus lauter nan gibt nan)."""
    with np.errstate(all="ignore"):
        m = np.nanmax(np.where(np.isnan(a), -np.inf, a), axis=1)
    return np.where(np.isinf(m), np.nan, m)


def loeschen(x, y, i: int):
    """Anpassung ohne Punkt i (float); `None`, wenn danach Sxx = 0 ist."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.arange(len(x)) != i
    if len(x) - 1 <= C.P or np.ptp(x[keep]) == 0:
        return None
    return fit(x[keep], y[keep])
