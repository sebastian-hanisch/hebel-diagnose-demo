"""Orakel-Tests: Hebel, Cook-Abstand und studentisierte Residuen gegen andere Rechenwege (Löschen und Neuanpassen, QR-Zerlegung, Verteilungsformeln, Aufzählung).

Orakel (jeweils ein anderer Weg als der Code der Demo):
  1. Hebel und Cook-D gegen Löschen und Neuanpassen (numpy.polyfit je ausgelassenem Punkt) in float, 1e-9 relativ; in Brüchen für Anscombe steht es in test_handbeispiele.py.
  2. Cook = t²/p · h/(1 − h) (Cook 1977, Gl. 8) und e = (1 − h)·(y − ŷ₍ᵢ₎).
  3. Hat-Matrix über QR (numpy.linalg.qr): h_i = Σ_j Q_ij², Spur gleich p.
  4. Beta-Orakel: bei normalen, gleich streuenden Fehlern ist t²/(n − p) ~ Beta(1/2, (n − p − 1)/2); daraus die erwartete Zahl Punkte über 4/n; t-Verteilung für |t*| > 3.
  5. Exakte Wahrscheinlichkeit „mindestens ein Kunde mit genau einem Container“ gegen Aufzählung aller Zuordnungen und gegen Simulation (Band).
  6. Generator (Export-Zweig) gegen feste Referenzwerte des Containerstroms der verweilzeit-demo (log-Median, berechnet mit deren `log_median_and_s`) und gegen die geschlossenen Momente.
Zufallsdatensätze kommen aus SplitMix64 (hdi_scenario), nicht aus numpy.random."""

import math
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
from scipy import stats

import hdi_algorithm as A
import hdi_constants as C
import hdi_exact as X
import hdi_measure as M
import hdi_scenario as S


def _datensaetze(R, n, welt="Verweilzeit", seed=77):
    d = S.draws(seed, R, n)
    y = S.werte(welt, d)
    return d.v[:, :n], y[:, :n]


def test_orakel_1_hebel_und_cook_gegen_loeschen_und_neuanpassen():
    x, y = _datensaetze(30, 12)
    r = A.fit_rows(x, y)
    worst = 0.0
    for j in range(30):
        b = np.polyfit(x[j], y[j], 1)
        yhat = np.polyval(b, x[j])
        s2 = np.sum((y[j] - yhat) ** 2) / (12 - 2)
        for i in range(12):
            m = np.arange(12) != i
            bi = np.polyfit(x[j][m], y[j][m], 1)
            di = np.sum((yhat - np.polyval(bi, x[j])) ** 2) / (2 * s2)
            worst = max(worst, abs(di - r.D[j, i]) / max(1.0, abs(di)))
    assert worst < 1e-9


def test_orakel_2_cook_gleich_t_quadrat_durch_p_mal_hebelquotient():
    x, y = _datensaetze(40, 20)
    r = A.fit_rows(x, y)
    assert np.allclose(r.D, r.t_int ** 2 / 2 * r.h / (1 - r.h), rtol=1e-10)


def test_orakel_2_residuum_ist_1_minus_h_mal_loeschresiduum():
    x, y = _datensaetze(20, 15)
    r = A.fit_rows(x, y)
    for j in range(20):
        for i in (0, 7, 14):
            m = np.arange(15) != i
            bi = np.polyfit(x[j][m], y[j][m], 1)
            dr = y[j][i] - np.polyval(bi, x[j][i])
            assert abs(r.e[j, i] - (1 - r.h[j, i]) * dr) < 1e-9 * max(1.0, abs(dr))


def test_orakel_3_hat_matrix_ueber_qr_gleich_formel_und_spur_p():
    x, y = _datensaetze(25, 18)
    r = A.fit_rows(x, y)
    for j in range(25):
        X_ = np.column_stack([np.ones(18), x[j]])
        Q, _ = np.linalg.qr(X_)
        h = (Q ** 2).sum(axis=1)
        assert np.allclose(h, r.h[j], atol=1e-12)
        assert abs(h.sum() - 2) < 1e-12


def test_orakel_3_extern_studentisiert_gegen_neuanpassung_ohne_den_punkt():
    """t*_i = e_i / (s_(i) √(1 − h_i)) mit s_(i) aus der Anpassung ohne Punkt i."""
    x, y = _datensaetze(15, 14)
    r = A.fit_rows(x, y)
    for j in range(15):
        for i in (0, 6, 13):
            m = np.arange(14) != i
            bi = np.polyfit(x[j][m], y[j][m], 1)
            s_i = math.sqrt(np.sum((y[j][m] - np.polyval(bi, x[j][m])) ** 2) / (14 - 1 - 2))
            assert abs(r.t_ext[j, i] - r.e[j, i] / (s_i * math.sqrt(1 - r.h[j, i]))) < 1e-8 * max(1.0, abs(r.t_ext[j, i]))


def test_orakel_4_beta_zahl_punkte_ueber_vier_durch_n_bei_normalen_fehlern():
    R, n = 3000, 30
    x, y = _datensaetze(R, n, "Normal", seed=1234)
    r = A.fit_rows(x, y)
    cnt = (r.D > 4 / n).sum(axis=1)
    xthr = 4 / n * 2 * (1 - r.h) / (r.h * (n - 2))               # t²/(n − p) muss diese Schwelle überschreiten
    erwartet = float(np.where(xthr >= 1.0, 0.0, stats.beta.sf(np.minimum(xthr, 1.0), 0.5, (n - 3) / 2)).sum(axis=1).mean())
    assert abs(erwartet - cnt.mean()) < 4 * cnt.std() / math.sqrt(R)
    assert 1.70 < erwartet < 1.80                               # die Zahl, die die README nennt (Beta-Orakel)


def test_orakel_4_t_verteilung_zahl_punkte_mit_betrag_t_ext_ueber_3():
    R, n = 3000, 30
    x, y = _datensaetze(R, n, "Normal", seed=4321)
    r = A.fit_rows(x, y)
    cnt = (np.abs(r.t_ext) > 3).sum(axis=1)
    erwartet = n * 2 * stats.t.sf(3, n - 3)
    assert abs(erwartet - cnt.mean()) < 4 * cnt.std() / math.sqrt(R)


def test_orakel_5_exakte_wahrscheinlichkeit_gegen_aufzaehlung():
    for kunden, n in ((3, 4), (3, 6), (4, 8)):
        ohne = sum(1 for a in product(range(kunden), repeat=n) if 1 not in [a.count(k) for k in range(kunden)])
        assert X.anzahl_ohne_einzelnen(n, kunden) == ohne
        assert X.p_mindestens_ein_einzelner(n, kunden) == 1 - F(ohne, kunden ** n)


@pytest.mark.parametrize("n", [24, 36, 60])
def test_orakel_5_exakte_wahrscheinlichkeit_gegen_simulation(n):
    R = 20000
    je = S.container_je_kunde(S.zuordnung(99, R, n))
    sim = float((je == 1).any(axis=1).mean())
    p = float(X.p_mindestens_ein_einzelner(n))
    assert abs(sim - p) < 4 * math.sqrt(p * (1 - p) / R) + 1e-12


def test_orakel_5_hebel_im_faktor_modell_gleich_eins_durch_zahl_der_container_des_kunden():
    for j, kunden in enumerate(S.zuordnung(5, 40, 30)):
        vorhanden = sorted(set(kunden.tolist()))
        Xd = np.zeros((30, len(vorhanden)))
        for i, k in enumerate(kunden):
            Xd[i, vorhanden.index(k)] = 1.0
        Q, _ = np.linalg.qr(Xd)
        h = (Q ** 2).sum(axis=1)
        groessen = np.array([(kunden == k).sum() for k in kunden])
        assert np.allclose(h, 1.0 / groessen, atol=1e-12)
        assert abs(h.sum() - len(vorhanden)) < 1e-12            # Spur = Zahl der Kunden


# --- Generator ---------------------------------------------------------------------------------------------------------------------------------

def test_orakel_6_log_median_gegen_referenzwerte_des_containerstroms():
    """Referenzwerte: log_median_and_s der verweilzeit-demo (Art Export, Vorlauf und Kunde wie unten), berechnet am 2026-11-16."""
    v = np.array([1.0, 2.5, 5.5, 7.25, 10.0])
    k = np.array([0, 3, 5, 9, 11])
    ref = [0.2434193031515662, 0.9791428369711116, 1.5962990606744254, 1.735634968501503, 2.2954391350404775]
    d_ohne_rauschen = S.verweilzeit(v, k, np.zeros(5))           # z = 0: d = exp(log-Median)
    assert np.allclose(np.log(d_ohne_rauschen), ref, atol=1e-12, rtol=0)
    assert abs(C.S_EXPORT - 0.25) < 1e-15                        # Streuung der log-Verweilzeit im Export laut Referenz: 0,25


def test_orakel_6_momente_der_verweilzeit_geschlossen_gegen_stichprobe():
    d = S.draws(31, 4000, 30)
    y = S.verweilzeit(d.v, d.k, d.z)
    resid = y - S.mean_d(d.v)
    assert abs(resid.mean()) < 4 * resid.std() / math.sqrt(resid.size)
    assert abs(resid.std() - S.SD_POOLED) < 0.05                 # geschlossen 1,640 (die Stichprobe von 120 000 Punkten streut um etwa 0,01)
    assert 1.62 < S.SD_POOLED < 1.66


def test_orakel_6_vorlauf_und_kunde_gleichverteilt():
    d = S.draws(8, 2000, 30)
    assert abs(d.v.mean() - 5.5) < 0.1 and d.v.min() >= 1.0 and d.v.max() < 10.0
    haeufigkeit = np.bincount(d.k.ravel(), minlength=12) / d.k.size
    assert np.max(np.abs(haeufigkeit - 1 / 12)) < 0.01
    assert abs(d.z.mean()) < 0.02 and abs(d.z.std() - 1) < 0.02


def test_orakel_messung_c_normal_stimmt_mit_beta_orakel():
    """Die in der README genannte Zahl: erwartete Punkte über 4/n bei normalen Fehlern (Beta-Orakel, bedingt auf die Hebel) gegen die Messung der Demo."""
    c = M.messung_c("Normal", 2024, 4000, 30)
    assert abs(c["mean_count_4n"] - 1.75) < 4 * c["se_count_4n"] + 0.02
