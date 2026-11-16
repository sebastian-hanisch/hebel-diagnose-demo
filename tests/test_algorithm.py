"""Kern von Hand: Hebel, Residuen, Cook-D und Schwellen an der Mini-Instanz (Handbeispiel 1) und an einem Datensatz mit bekannter Lösung (kleinste Quadrate auf exakten Daten)."""

import numpy as np
import pytest

import hdi_algorithm as A


def test_mini_instanz_von_hand(mini):
    x, y = mini
    r = A.fit([float(v) for v in x], [float(v) for v in y])
    assert r.b0[0] == pytest.approx(2.0) and r.b1[0] == pytest.approx(1.0)
    assert r.e[0] == pytest.approx([-1.0, 0.0, 1.0, 0.0], abs=1e-12)
    assert r.h[0, :3] == pytest.approx([1 / 3] * 3) and r.h[0, 3] == pytest.approx(1.0)
    assert r.s2[0] == pytest.approx(1.0)                        # SSE 2, n − p = 2
    assert r.D[0, 0] == pytest.approx(3 / 8) and r.D[0, 1] == pytest.approx(0.0, abs=1e-12) and r.D[0, 2] == pytest.approx(3 / 8)
    assert np.isnan(r.D[0, 3])                                   # Hebel 1: undefiniert, nie inf


def test_exakte_gerade_hat_residuum_null_und_cook_null():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = A.fit(x, 3.0 + 2.0 * x)
    assert r.b0[0] == pytest.approx(3.0) and r.b1[0] == pytest.approx(2.0)
    assert np.allclose(r.e, 0, atol=1e-12)
    assert np.all(np.nan_to_num(r.D) < 1e-12)


def test_hebel_summe_ist_p_und_zwischen_1_durch_n_und_1():
    rng = np.random.default_rng(3)                               # nur Testdaten; die App benutzt SplitMix64
    x = rng.uniform(0, 10, (20, 15))
    y = rng.normal(size=(20, 15))
    r = A.fit_rows(x, y)
    assert np.allclose(r.h.sum(axis=1), 2.0)
    assert np.all(r.h >= 1 / 15 - 1e-12) and np.all(r.h <= 1 + 1e-12)


def test_schwellen_von_hand():
    t = A.schwellen(11)
    assert t["hebel"] == pytest.approx(4 / 11) and t["cook_4n"] == pytest.approx(4 / 11)          # bei p = 2 dieselbe Zahl
    assert t["cook_f50"] == pytest.approx(0.7494, abs=5e-4)                                      # F(2, 9; 0,5), Handwert aus Cook-Tabellen / scipy
    assert t["cook_1"] == 1.0 and t["t3"] == 3.0
    assert A.schwellen(31)["hebel"] == pytest.approx(4 / 31)


def test_alarme_an_von_hand_b_nur_der_sechste_punkt():
    """Von Hand B mit y6 = 0 (n = 6): Hebel 19/24 > 2p/n = 2/3, Cook-D 152/35 = 4,34 > 4/n = 2/3 und > 1, die fünf anderen liegen darunter (höchstens 0,30)."""
    x = [-2.0, -1.0, 0.0, 1.0, 2.0, 6.0]
    y = [-1.0, -2.0, 0.0, 2.0, 1.0, 0.0]
    fl = A.alarme(A.fit(x, y))
    for name in ("hebel", "cook_4n", "cook_1"):
        assert fl[name][0].tolist() == [False, False, False, False, False, True], name


def test_alarme_zaehlen_undefiniertes_cook_nicht(mini):
    x, y = mini
    fl = A.alarme(A.fit([float(v) for v in x], [float(v) for v in y]))
    assert not fl["cook_4n"][0, 3] and not fl["cook_1"][0, 3]          # Punkt 4: D = 0/0 zählt nie als Alarm


def test_max_ohne_nan_und_zeile_aus_lauter_nan():
    a = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    m = A.max_ohne_nan(a)
    assert m[0] == 3.0 and np.isnan(m[1])


def test_loeschen_gibt_none_wenn_steigung_nicht_bestimmt():
    x = [8, 8, 8, 19.0]
    y = [1.0, 2.0, 3.0, 6.0]
    assert A.loeschen(x, y, 3) is None
    assert A.loeschen(x, y, 0) is not None
