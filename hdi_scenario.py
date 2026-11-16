"""Szenario: Container-Verweilzeit (Export-Zweig) und eine zweite Welt „Normal“ mit gleichem Mittel und konstanter Streuung; Zufallsströme aus SplitMix64.

Herkunft der Formeln: der Containerstrom der `verweilzeit-demo` (Export-Zweig: Median 0,6 + 0,85·Vorlauf Tage, Kundeneffekt u_k mit Faktor 0,3, Streuung der log-Verweilzeit 0,25;
die zwölf Kundeneffekte als feste Liste in `hdi_constants.py`). Hier **nachgebaut**, ohne Import aus der anderen Demo, mit dem plattformfesten Zufallsstrom aus `hdi_rng.py`:
d = (0,6 + 0,85·v) · exp(0,3·u_Kunde) · exp(0,25·z), z ~ N(0; 1), v ~ Gleich(1; 10), Kunde gleichverteilt (statt log(0,6 + 0,85·v) + … wird das Produkt gerechnet:
gleiche Zahl, aber ohne Logarithmus der C-Bibliothek). Der Vorlauf des Zusatzpunkts außerhalb von 1 bis 10 Tagen ist **gesetzt** (Schiff verschoben), nicht Teil des Containerstroms.

Zufallsstrom: ein zählerbasierter Strom je Zufalls-Seed; Datensatz j belegt den Abschnitt j·4·(n + 2) bis (j + 1)·4·(n + 2) (je Punkt vier Ausgaben: Vorlauf, Kunde, zwei für die
Normalzahl). Die ersten R Datensätze sind deshalb für jedes R dieselben (Präfixeigenschaft), und die Werte eines Datensatzes hängen nicht von der Zahl der übrigen Datensätze ab."""

from dataclasses import dataclass

import numpy as np

import hdi_constants as C
import hdi_rng as RNG

UK = np.array(C.UK)
EK = np.array([RNG.exact_exp(C.U_SCALE_EXPORT * float(u)) for u in C.UK])                      # exp(0,3·u_k), plattformfest
C_MEAN = RNG.exact_exp(C.S_EXPORT ** 2 / 2.0) * float(EK.mean())                                # E[d | v] = C_MEAN·(0,6 + 0,85·v)


def _sd_pooled() -> float:
    """Gepoolte Standardabweichung von d − E[d | v] im Containerstrom, geschlossen: √(E[m²]·(E[EK²]·e^{S²} − c²)), m = 0,6 + 0,85·v mit v ~ Gleich(1; 10)."""
    mean_m = C.M0 + C.M1 * (C.V_LO + C.V_HI) / 2.0
    var_m = (C.M1 * (C.V_HI - C.V_LO)) ** 2 / 12.0
    e_m2 = var_m + mean_m ** 2
    e_ek2 = float(np.mean([RNG.exact_exp(2.0 * C.U_SCALE_EXPORT * float(u)) for u in C.UK]))
    return float(np.sqrt(e_m2 * (e_ek2 * RNG.exact_exp(2.0 * C.S_EXPORT ** 2) - C_MEAN ** 2)))


SD_POOLED = _sd_pooled()


def mean_d(v):
    """Erwartete Verweilzeit bei Vorlauf v (Tage)."""
    return C_MEAN * (C.M0 + C.M1 * np.asarray(v, dtype=float))


@dataclass(frozen=True)
class Draws:
    """Zufallsgrößen je Datensatz: Vorlauf v, Kunde k, Normalzahl z; Form (R, n + 2). Spalten 0 bis n − 1 sind die Container, n und n + 1 die Reservepunkte."""
    n: int
    v: np.ndarray
    k: np.ndarray
    z: np.ndarray


def draws(seed: int, R: int, n: int) -> Draws:
    cols = n + C.EXTRA_PUNKTE
    u = RNG.splitmix_array(seed, R * cols * 4).reshape(R, cols, 4)
    v = C.V_LO + (C.V_HI - C.V_LO) * RNG.to_uniform(u[..., 0])
    k = RNG.below_array(u[..., 1], C.K_KUNDEN)
    z = RNG.normal_from_uniforms(RNG.to_uniform(u[..., 2]), RNG.to_uniform(u[..., 3]))
    return Draws(n, v, k, z)


def verweilzeit(v, k, z):
    """Export-Container: d = (0,6 + 0,85·v)·exp(0,3·u_k)·exp(0,25·z)."""
    return (C.M0 + C.M1 * np.asarray(v, dtype=float)) * EK[k] * RNG.exp_array(C.S_EXPORT * np.asarray(z, dtype=float))


def normal_welt(v, z):
    """Welt „Normal“: gleiches Mittel, konstante Standardabweichung SD_POOLED, normale Fehler."""
    return mean_d(v) + SD_POOLED * np.asarray(z, dtype=float)


def werte(welt: str, d: Draws, v=None) -> np.ndarray:
    """Verweilzeit je Punkt (Form (R, n + 2)) in der gewählten Welt; `v` ersetzt den Vorlauf (z. B. der gesetzte Vorlauf des Zusatzpunkts)."""
    vv = d.v if v is None else v
    if welt == "Verweilzeit":
        return verweilzeit(vv, d.k, d.z)
    if welt == "Normal":
        return normal_welt(vv, d.z)
    raise ValueError(f"Unbekannte Welt: {welt}")


def werte_punkt(welt: str, d: Draws, col: int, v) -> np.ndarray:
    """Verweilzeit des Punkts in Spalte `col` bei gesetztem Vorlauf v (Zahl oder Feld über die Datensätze): Zusatzpunkte mit gesetztem Vorlauf."""
    vv = np.full(d.v.shape[0], float(v)) if np.isscalar(v) else np.asarray(v, dtype=float)
    if welt == "Verweilzeit":
        return verweilzeit(vv, d.k[:, col], d.z[:, col])
    if welt == "Normal":
        return normal_welt(vv, d.z[:, col])
    raise ValueError(f"Unbekannte Welt: {welt}")


def zuordnung(seed: int, R: int, n: int) -> np.ndarray:
    """Kunde je Container, Form (R, n): jeder der zwölf Kunden mit gleicher Wahrscheinlichkeit."""
    return RNG.below_array(RNG.splitmix_array(seed, R * n), C.K_KUNDEN).reshape(R, n)


def container_je_kunde(kunden: np.ndarray) -> np.ndarray:
    """Zahl der Container je Kunde, Form (R, 12)."""
    return np.stack([(kunden == j).sum(axis=1) for j in range(C.K_KUNDEN)], axis=1)
