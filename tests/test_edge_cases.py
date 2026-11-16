"""Randfälle, an denen die Diagnose erfahrungsgemäß bricht (DEMO-PLAYBOOK §12): zu wenige Punkte, alle x gleich, Hebel genau 1, Zusatzpunkt mit Faktor 0, Wurzel von n − p − t² ≤ 0,
Gleichstand in Cook-D, Kunde ohne Container. Nie `inf` oder `nan` in der Anzeige."""

from fractions import Fraction as F

import numpy as np
import pytest

import hdi_algorithm as A
import hdi_constants as C
import hdi_evaluation as ev
import hdi_exact as X
import hdi_scenario as S


def test_n_gleich_2_meldung_statt_zahl():
    """Mit zwei Punkten ist n − p = 0: s² ist nicht definiert; beide Wege melden es statt eine Zahl zu liefern."""
    with pytest.raises(ValueError, match="nicht definiert"):
        A.fit([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="nicht definiert"):
        X.fit([F(0), F(1)], [F(0), F(1)])


def test_n_gleich_3_ist_der_kleinste_bestimmte_fall():
    r = A.fit([0.0, 1.0, 3.0], [0.0, 1.0, 1.0])
    assert np.isfinite(r.s2[0]) and np.all(r.h < 1.0) and abs(r.h.sum() - 2) < 1e-12
    f = X.fit([F(0), F(1), F(3)], [F(0), F(1), F(1)])
    assert f["n"] == 3 and sum(f["h"]) == 2


def test_alle_x_gleich_steigung_nicht_bestimmt():
    assert X.fit([F(2)] * 4, [F(1), F(2), F(3), F(4)]) is None
    r = A.fit([2.0] * 4, [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(r.b1[0]) and not np.isinf(r.D).any()
    assert A.loeschen([2.0, 2.0, 2.0, 5.0], [1.0, 2.0, 3.0, 4.0], 3) is None


def test_hebel_nahe_1_gilt_als_1_und_cook_ist_nan_nie_inf():
    x = np.array([0.0] * 9 + [1.0e6])
    y = np.arange(10.0) ** 1.5
    r = A.fit(x, y)
    assert r.h[0, 9] > A.H_EINS                                  # Hebel 1 bis auf Rundung
    assert np.isnan(r.D[0, 9]) and not np.isinf(r.D).any() and not np.isinf(r.t_ext).any()


def test_wurzel_von_n_minus_p_minus_t_quadrat_nicht_positiv_gibt_nan_statt_fehler():
    """Liegen alle anderen Punkte genau auf einer Geraden, übernimmt der letzte die ganze Residuenquadratsumme: t² = n − p, die Wurzel von t* hat Radikand 0."""
    r = A.fit([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 0.0])
    assert not np.isinf(r.t_ext).any()
    assert np.isnan(r.t_ext[0, 3]) or abs(r.t_ext[0, 3]) > 1e6   # Rundung kann Radikand 0 oder ein winziges Plus treffen; nie Fehler oder inf


def test_zusatzpunkt_mit_extremen_reglerwerten_gibt_endliche_zahlen():
    for v, f, n in ((C.V_MIN, C.F_MIN, C.N_MIN), (C.V_MAX, C.F_MAX, C.N_MAX), (C.V_MAX, 0.0, C.N_MIN), (5.5, 1.0, C.N_MAX)):
        a = ev.analyse_zusatz(ev.Settings(seed=3, n=n, v=v, f=f))
        for wert in (a.h, a.e_s, a.t_int, a.t_ext, a.d, a.rel, a.se_ratio):
            assert np.isfinite(wert), (v, f, n, wert)
        assert len(a.kurve_v) == len(a.kurve_d) and np.isfinite(a.kurve_d).all()


def test_gleichstand_in_cook_d_wird_gemeldet_und_nach_kleinstem_index_aufgeloest():
    f = X.anscombe("II")
    assert X.gleichstand(f, "Größtes Cook-D") == [5, 7]
    assert X.waehle_punkt(f, "Größtes Cook-D") == 5
    ex = ev.analyse_anscombe("II", "Größtes Cook-D")
    assert ex.gleich == (5, 7) and "Gleichstand: die Punkte 6 und 8" in ev.verdict_anscombe("II", ex)[1]


def test_punktregel_bei_hebel_1_waehlt_den_hebelpunkt_und_meldet_nicht_bestimmte_gerade():
    ex = ev.analyse_anscombe("IV", "Größter Hebel")
    assert ex.punkt == 7 and ex.ohne is None and ex.loesch_res is None
    ex = ev.analyse_anscombe("IV", "Größtes Cook-D")
    assert ex.punkt == 3                                         # Punkt 8 hat kein D (0/0), das größte definierte steht bei Punkt 4


def test_unbekannte_regel_und_welt_werden_abgewiesen():
    with pytest.raises(ValueError):
        X.waehle_punkt(X.hand_a(), "Zufällig")
    with pytest.raises(ValueError):
        S.werte("Unsinn", S.draws(1, 2, 5))


def test_kunde_ohne_container_faellt_aus_dem_faktor_modell():
    """12 Container auf 12 Kunden: im Mittel fehlen Kunden ganz; fehlt einer, hat er keine Spalte (Spur = Zahl der vorhandenen Kunden)."""
    je = S.container_je_kunde(S.zuordnung(5, 200, 12))
    assert ((je == 0).any(axis=1)).mean() > 0.99                 # 12 Container, 12 Kunden: fast immer fehlt einer
    assert (je.sum(axis=1) == 12).all()


def test_exakte_wahrscheinlichkeit_in_den_randfaellen():
    assert X.p_mindestens_ein_einzelner(0) == 0                  # kein Container, kein Kunde mit genau einem
    assert X.p_mindestens_ein_einzelner(1) == 1                  # ein Container: sein Kunde hat genau einen
    assert X.p_mindestens_ein_einzelner(2, 1) == 0               # ein Kunde, zwei Container
    assert X.anzahl_ohne_einzelnen(0, 12) == 1


def test_nenner_null_zweige_der_texte():
    """Ersparnis-ähnliche Quotienten: Steigung der Basis 0 gibt nan statt Fehler (kein Absturz in der Anzeige)."""
    r = A.fit([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0])
    assert np.isfinite(r.b1[0]) and abs(r.b1[0]) < 1e-12
    assert ev.de(0.0) == "0,00" and ev.pct(0.0) == "0,0 %"
