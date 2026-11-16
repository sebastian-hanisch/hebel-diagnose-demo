"""Eigenschaftstests (metamorph): Beziehungen, die für Hebel, Residuen und Cook-Abstand gelten müssen, ohne dass man das richtige Ergebnis kennt.

Je Eigenschaft 30 Zufallsdatensätze (SplitMix64, fester Zufalls-Seed). Jede Eigenschaft ist eine Aussage über die Gerade nach kleinsten Quadraten, nicht über das heutige Verhalten des Codes."""

import numpy as np

import hdi_algorithm as A
import hdi_scenario as S

R, N = 30, 14


def _daten(seed=2026):
    d = S.draws(seed, R, N)
    return d.v[:, :N], S.werte("Verweilzeit", d)[:, :N]


def test_spur_ist_p_hebel_zwischen_0_und_1_mittel_p_durch_n():
    x, y = _daten()
    r = A.fit_rows(x, y)
    assert np.allclose(r.h.sum(axis=1), 2.0, atol=1e-12)
    assert (r.h > 0).all() and (r.h <= 1 + 1e-12).all()
    assert abs(r.h.mean() - 2 / N) < 1e-12


def test_hebel_haengt_nicht_an_y():
    x, y = _daten()
    r1 = A.fit_rows(x, y)
    r2 = A.fit_rows(x, y[::-1] * 3.0 + 7.0)
    assert np.array_equal(r1.h, r2.h)


def test_verschieben_aller_y_aendert_residuen_hebel_und_cook_nicht():
    x, y = _daten()
    a, b = A.fit_rows(x, y), A.fit_rows(x, y + 5.0)
    assert np.allclose(a.e, b.e, atol=1e-9) and np.allclose(a.h, b.h) and np.allclose(a.D, b.D, rtol=1e-8, atol=1e-10)
    assert np.allclose(b.b0, a.b0 + 5.0, atol=1e-9) and np.allclose(a.b1, b.b1, atol=1e-10)


def test_skalieren_von_y_skaliert_residuen_laesst_cook_und_t_unveraendert():
    x, y = _daten()
    a, b = A.fit_rows(x, y), A.fit_rows(x, 2.5 * y)
    assert np.allclose(b.e, 2.5 * a.e, atol=1e-9)
    assert np.allclose(a.D, b.D, rtol=1e-8, atol=1e-10)
    assert np.allclose(a.t_int, b.t_int, atol=1e-9) and np.allclose(a.t_ext, b.t_ext, atol=1e-8)


def test_skalieren_und_verschieben_von_x_aendert_hebel_und_cook_nicht():
    x, y = _daten()
    a, b = A.fit_rows(x, y), A.fit_rows(4.0 * x - 3.0, y)
    assert np.allclose(a.h, b.h, atol=1e-12) and np.allclose(a.D, b.D, rtol=1e-8, atol=1e-10)
    assert np.allclose(a.b1, 4.0 * b.b1, rtol=1e-9)


def test_spiegeln_von_x_laesst_hebel_und_cook_gleich():
    x, y = _daten()
    a, b = A.fit_rows(x, y), A.fit_rows(-x, y)
    assert np.allclose(a.h, b.h, atol=1e-12) and np.allclose(a.D, b.D, rtol=1e-8, atol=1e-10)
    assert np.allclose(a.b1, -b.b1, rtol=1e-9)


def test_vertauschen_der_punkte_permutiert_hebel_residuen_cook():
    x, y = _daten()
    perm = np.random.default_rng(1).permutation(N)               # nur Testdaten: eine feste Permutation
    a, b = A.fit_rows(x, y), A.fit_rows(x[:, perm], y[:, perm])
    assert np.allclose(a.h[:, perm], b.h, atol=1e-12) and np.allclose(a.e[:, perm], b.e, atol=1e-9) and np.allclose(a.D[:, perm], b.D, rtol=1e-8, atol=1e-10)


def test_zusatzpunkt_auf_der_geraden_aendert_gerade_nicht_und_senkt_den_standardfehler():
    x, y = _daten()
    a = A.fit_rows(x, y)
    xv = np.full((R, 1), 25.0)                                   # weit außerhalb: Hebel hoch
    yv = (a.b0 + a.b1 * 25.0)[:, None]
    b = A.fit_rows(np.concatenate([x, xv], axis=1), np.concatenate([y, yv], axis=1))
    assert np.allclose(a.b0, b.b0, atol=1e-8) and np.allclose(a.b1, b.b1, atol=1e-9)
    assert np.allclose(b.e[:, N], 0, atol=1e-8) and np.allclose(b.D[:, N], 0, atol=1e-10)
    assert (b.se1 < a.se1).all()                                 # ein guter Hebelpunkt macht die Steigung genauer (s² sinkt nicht nur, sondern Sxx wächst)


def test_hebel_eines_zusatzpunkts_waechst_mit_dem_abstand_vom_mittel():
    x, y = _daten()
    hs = []
    for v in (6.0, 10.0, 15.0, 30.0):
        r = A.fit_rows(np.concatenate([x, np.full((R, 1), v)], axis=1), np.concatenate([y, np.full((R, 1), 5.0)], axis=1))
        hs.append(r.h[:, N])
    for kleiner, groesser in zip(hs[:-1], hs[1:]):
        assert (groesser > kleiner).all()


def test_gleiche_seeds_gleiche_datensaetze_und_praefixeigenschaft():
    d1, d2 = S.draws(5, 10, 14), S.draws(5, 10, 14)
    assert np.array_equal(d1.v, d2.v) and np.array_equal(d1.z, d2.z) and np.array_equal(d1.k, d2.k)
    gross = S.draws(5, 40, 14)
    assert np.array_equal(gross.v[:10], d1.v) and np.array_equal(gross.z[:10], d1.z)     # die ersten 10 Datensätze sind für jedes R dieselben


def test_residuen_sind_orthogonal_zu_den_spalten():
    x, y = _daten()
    r = A.fit_rows(x, y)
    assert np.allclose(r.e.sum(axis=1), 0, atol=1e-9) and np.allclose((r.e * x).sum(axis=1), 0, atol=1e-7)
