"""Handbeispiele in Brüchen (fractions): jede Zahl ist von Hand gerechnet, nicht aus dem Code der Demo erzeugt; zusätzlich gleicht jeder Test die Rechnung der Demo (hdi_exact) Bruch gleich Bruch ab.

Konvention: p = 2 (Achsenabschnitt und Steigung), Hebel h_i = 1/n + (x_i − x̄)²/Sxx, Cook-D = e² h / (p s² (1 − h)²) mit s² = SSE/(n − p) (Cook 1977, Gl. 4 und 8).
Die kleine Rechnung hier ist absichtlich ein zweiter Weg (eigene Zeilen statt hdi_exact): sie rechnet die Gerade aus den Normalgleichungen."""

from fractions import Fraction as F
from itertools import product

import pytest

import hdi_exact as X


def gerade(x, y):
    """Normalgleichungen für zwei Unbekannte, gelöst per Cramer-Regel (anderer Weg als in hdi_exact: dort Sxy/Sxx)."""
    n = len(x)
    sx, sy = sum(x), sum(y)
    sxx, sxy = sum(a * a for a in x), sum(a * b for a, b in zip(x, y))
    det = n * sxx - sx * sx
    if det == 0:
        return None
    b1 = (n * sxy - sx * sy) / det
    b0 = (sy - b1 * sx) / n
    e = [b - (b0 + b1 * a) for a, b in zip(x, y)]
    # Hebel aus der Hat-Matrix h_i = (1, x_i) (X'X)^-1 (1, x_i)'
    h = [(sxx - 2 * sx * a + n * a * a) / det for a in x]
    sse = sum(v * v for v in e)
    return b0, b1, e, h, sse


def cook(e, h, sse, n):
    s2 = sse / (n - 2)
    return [None if hi == 1 else ei ** 2 * hi / (2 * s2 * (1 - hi) ** 2) for ei, hi in zip(e, h)]


# --- Handbeispiel 1: ein Kunde mit einem Container -----------------------------------------------------------------------------------------------

X1 = [F(0), F(0), F(0), F(4)]
Y1 = [F(1), F(2), F(3), F(6)]


def test_h1_gerade_hebel_residuen_cook():
    b0, b1, e, h, sse = gerade(X1, Y1)
    assert (b0, b1) == (2, 1)                                   # y = 2 + x
    assert e == [-1, 0, 1, 0]
    assert h == [F(1, 3), F(1, 3), F(1, 3), F(1)]               # 1/Zahl der Container des Kunden
    assert sum(h) == 2                                          # Spur der Hat-Matrix = p
    assert sse == 2
    assert cook(e, h, sse, 4) == [F(3, 8), 0, F(3, 8), None]


def test_h1_demo_gleich_handrechnung():
    f = X.hand_a()
    assert f["b0"] == 2 and f["b1"] == 1
    assert f["h"] == [F(1, 3), F(1, 3), F(1, 3), F(1)]
    assert f["D"] == [F(3, 8), 0, F(3, 8), None]
    assert f["e"] == [-1, 0, 1, 0]


def test_h1_ohne_den_vierten_punkt_ist_die_steigung_nicht_bestimmt():
    assert gerade(X1[:3], Y1[:3]) is None
    assert X.fit(X1[:3], Y1[:3]) is None


def test_h1_loeschen_von_punkt_1():
    b0, b1, *_ = gerade(X1[1:], Y1[1:])
    assert (b0, b1) == (F(5, 2), F(7, 8))
    ohne = X.loo(X1, Y1, 0)
    assert ohne["b0"] == F(5, 2) and ohne["b1"] == F(7, 8)
    f = X.hand_a()
    assert X.cook_by_deletion(f, 0) == F(3, 8) == f["D"][0]
    dr = X.loeschresiduum(f, 0)
    assert dr == F(-3, 2) and f["e"][0] == (1 - f["h"][0]) * dr == -1


# --- Handbeispiel 2: derselbe sechste Punkt, einmal auf der Geraden, einmal daneben ---------------------------------------------------------------

XB = [F(-2), F(-1), F(0), F(1), F(2)]
YB = [F(-1), F(-2), F(0), F(2), F(1)]


def test_h2_basis():
    b0, b1, e, h, sse = gerade(XB, YB)
    assert (b0, b1) == (0, F(4, 5))
    assert e == [F(3, 5), F(-6, 5), 0, F(6, 5), F(-3, 5)]
    assert sse == F(18, 5) and sse / 3 == F(6, 5)
    assert h == [F(3, 5), F(3, 10), F(1, 5), F(3, 10), F(3, 5)]
    assert (sse / 3) / 10 == F(3, 25)                           # SE(b1)^2 = s^2/Sxx mit Sxx = 10


def test_h2a_sechster_punkt_auf_der_geraden():
    f, base = X.hand_b(24)
    b0, b1, e, h, sse = gerade(XB + [F(6)], YB + [F(24, 5)])
    assert (b0, b1) == (0, F(4, 5))
    assert h[5] == F(19, 24) and e[5] == 0
    assert cook(e, h, sse, 6)[5] == 0
    assert sse / 4 == F(9, 10) and (sse / 4) / 40 == F(9, 400)
    assert f["se1_sq"] == F(9, 400) and base["se1_sq"] == F(3, 25)
    assert f["se1_sq"] / base["se1_sq"] == F(3, 16)             # Quadrat des Verhältnisses der Standardfehler
    assert abs(float(f["se1_sq"] / base["se1_sq"]) ** 0.5 - 0.4330127018922193) < 1e-12


def test_h2b_sechster_punkt_daneben():
    b0, b1, e, h, sse = gerade(XB + [F(6)], YB + [F(0)])
    assert (b0, b1) == (F(-1, 5), F(1, 5))                      # Steigung von 4/5 auf 1/5
    assert e == [F(-2, 5), F(-8, 5), F(1, 5), F(2), F(4, 5), F(-1)]
    assert sorted((abs(v) for v in e), reverse=True) == [2, F(8, 5), 1, F(4, 5), F(2, 5), F(1, 5)]   # |e6| = 1 steht auf Platz 3
    assert sse == F(42, 5) and sse / 4 == F(21, 10)
    d = cook(e, h, sse, 6)
    assert d[5] == F(152, 35)                                   # = (19/24) / (2 · 21/10 · 25/576)
    assert round(float(d[5]), 3) == 4.343
    assert max(d[:5]) < F(1, 2) and float(max(d[:5])) < 0.31


def test_h2b_residuum_ist_der_anteil_1_minus_h_der_abweichung_ohne_den_punkt():
    f, base = X.hand_b(0)
    dr = f["y"][5] - (base["b0"] + base["b1"] * f["x"][5])       # y6 − Vorhersage der ersten fünf = −24/5
    assert dr == F(-24, 5)
    assert f["e"][5] == (1 - F(19, 24)) * dr == -1
    assert X.loeschresiduum(f, 5) == dr


def test_h2b_cook_formel_gleich_t_quadrat_durch_p_mal_hebel_quotient_und_loeschen():
    f, base = X.hand_b(0)
    h, e, s2 = f["h"][5], f["e"][5], f["s2"]
    t2 = e ** 2 / (s2 * (1 - h))                                 # intern studentisiert zum Quadrat
    assert f["D"][5] == t2 / 2 * h / (1 - h)                     # Cook Gl. 4 und 8
    assert X.cook_by_deletion(f, 5) == f["D"][5]                 # Löschen und Neuanpassen
    assert (f["b1"] - base["b1"]) ** 2 / base["se1_sq"] == 3     # Verschiebung der Steigung: √3 Standardfehler der Basis


def test_h2_demo_gleich_handrechnung():
    f, _ = X.hand_b(0)
    assert f["b0"] == F(-1, 5) and f["b1"] == F(1, 5)
    assert f["D"][5] == F(152, 35) and f["h"][5] == F(19, 24)


# --- Handbeispiel 3: Konvention ------------------------------------------------------------------------------------------------------------------

def test_h3_mittel_der_hebel_und_schwellen():
    f, _ = X.hand_b(0)
    assert sum(f["h"]) / 6 == F(2, 6)                            # p/n mit p inklusive Achsenabschnitt
    assert 2 * F(2, 6) == F(4, 6)                                # 2p/n; bei p = 2 dieselbe Zahl wie 4/n
    assert [i for i in range(6) if f["h"][i] > F(4, 6)] == [5]   # nur der sechste Punkt liegt über 2p/n


# --- Handbeispiel 4: wie oft hat ein Kunde genau einen Container? ----------------------------------------------------------------------------------

def test_h4_aufzaehlung_k3_n4():
    ohne = sum(1 for a in product(range(3), repeat=4) if 1 not in [a.count(k) for k in range(3)])
    assert ohne == 21                                            # (4,0,0) dreimal und (2,2,0) achtzehnmal
    assert F(81 - ohne, 81) == F(20, 27) == X.p_mindestens_ein_einzelner(4, 3)


def test_h4_spur_der_hebel_ist_die_zahl_der_vorhandenen_kunden():
    for a in product(range(3), repeat=4):
        groessen = [a.count(k) for k in range(3)]
        assert sum(F(1, groessen[k]) for k in a) == sum(1 for g in groessen if g > 0)


# --- Handbeispiel 5: Anscombe-Zahlen --------------------------------------------------------------------------------------------------------------

def test_h5_satz_iv_hebel_genau_1():
    f = X.anscombe("IV")
    assert f["xb"] == 9 and f["sxx"] == 110
    assert f["h"][7] == 1 and f["e"][7] == 0 and f["D"][7] is None
    assert all(f["h"][i] == F(1, 10) for i in range(11) if i != 7)
    xs, ys = X.parse("8 8 8 8 8 8 8 19 8 8 8"), X.parse("658 576 771 884 847 704 525 1250 556 791 689", 100)
    assert X.fit(xs[:7] + xs[8:], ys[:7] + ys[8:]) is None       # ohne Punkt 8: Sxx = 0


def test_h5_satz_iii_cook_und_hebel():
    f = X.anscombe("III")
    assert round(float(f["D"][2]), 3) == 1.393
    assert round(float(f["h"][2]), 4) == 0.2364 and round(float(max(f["h"])), 4) == 0.3182
    assert sum(f["h"]) == 2


def test_h5_satz_i_ueber_vier_durch_n():
    f = X.anscombe("I")
    assert round(float(max(f["D"])), 3) == 0.489 and max(f["D"]) > F(4, 11)


def test_h5_satz_ii_gleichstand_der_punkte_6_und_8():
    f = X.anscombe("II")
    assert f["D"][5] == f["D"][7] and round(float(f["D"][5]), 3) == 0.808
    assert X.gleichstand(f, "Größtes Cook-D") == [5, 7]
    assert X.waehle_punkt(f, "Größtes Cook-D") == 5               # Gleichstand: der kleinere Index


def test_h5_f_quantil_marke():
    stats = pytest.importorskip("scipy.stats")
    assert abs(stats.f.ppf(0.5, 2, 9) - 0.7494) < 5e-4


def test_anscombe_alle_vier_saetze_dieselbe_gerade():
    for satz in ("I", "II", "III", "IV"):
        f = X.anscombe(satz)
        assert abs(float(f["b0"]) - 3.00) < 0.01 and abs(float(f["b1"]) - 0.50) < 0.001
        sst = sum((v - f["yb"]) ** 2 for v in f["y"])
        assert 0.666 <= float(1 - f["sse"] / sst) <= 0.667
        assert sum(f["h"]) == 2
