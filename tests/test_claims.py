"""Claims-Tests: jede Zahl der Befunde B1 bis B9 und der Berichtigungen der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung überall: einfache Gerade d = b0 + b1·v (OLS), Export-Container (Vorlauf Gleich(1; 10)), n = 30 Container plus Zusatzpunkt, Zähleinheit Datensatz, sofern nicht anders genannt.
Plattform: Brüche exakt; die Tabellen der README stehen Zeile für Zeile in `data/hdi_frozen.json` (die Datei wird in test_frozen.py mit der Probezelle neu gerechnet);
Zahlen der Anzeige werden aus der Datei gerundet und im README-Text gesucht (`_shows`). Zufallsstrom: SplitMix64, auf jeder Plattform dieselbe Folge."""

import math
import sys
from fractions import Fraction as F
from pathlib import Path

from scipy import stats

import hdi_algorithm as A
import hdi_constants as C
import hdi_evaluation as ev
import hdi_exact as X
import hdi_frozen as Z
import hdi_scenario as S

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hdi_readme_tabellen as T  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
FROZEN = Z.laden()


def _has(*texte):
    for t in texte:
        assert t in README, t


def _shows(value, text, nd=None):
    """Die Rechnung rundet auf den README-Text, und der Text steht in der README."""
    if nd is not None:
        assert ev.de(value, nd) == text, (value, text)
    _has(text)


def _row(welt, v, f):
    return Z.zeile(FROZEN, welt, v, f)


# --- B1, B2: Handbeispiel 2 ---------------------------------------------------------------------------------------------------------------------

def test_b1_b2_von_hand_b():
    f, base = X.hand_b(0)
    assert f["h"][5] == F(19, 24) and f["b1"] == F(1, 5) and f["b0"] == F(-1, 5) and base["b1"] == F(4, 5)
    assert f["e"][5] == -1 and f["D"][5] == F(152, 35) and round(float(max(f["D"][:5])), 2) == 0.30
    assert sorted((abs(v) for v in f["e"]), reverse=True).index(1) == 2                      # Platz 3 der Beträge
    assert base["h"] == [F(3, 5), F(3, 10), F(1, 5), F(3, 10), F(3, 5)] and base["e"] == [F(3, 5), F(-6, 5), 0, F(6, 5), F(-3, 5)]
    assert F(5, 24) == 1 - F(19, 24) and X.loeschresiduum(f, 5) == F(-24, 5) and f["e"][5] == F(5, 24) * F(-24, 5)
    _has("19/24", "y = −1/5 + 1/5·x", "**auf Platz 3**", "(2, 8/5, 1, 4/5, 2/5, 1/5)", "152/35 = 4,34", "höchstens 0,30", "5/24", "Hebel 3/5, 3/10, 1/5, 3/10, 3/5",
         "Residuen 3/5, −6/5, 0, 6/5, −3/5", "x = −2 bis 2 mit y = −1, −2, 0, 2, 1")
    g, b0 = X.hand_b(24)
    assert g["b1"] == F(4, 5) and g["b0"] == 0 and g["e"][5] == 0 and g["D"][5] == 0
    assert g["s2"] == F(9, 10) and g["se1_sq"] == F(9, 400) and b0["se1_sq"] == F(3, 25)
    _shows(math.sqrt(g["se1_sq"] / b0["se1_sq"]), "0,433", 3)
    _has("(s² 9/10, SE² 9/400 gegen 3/25)", "(3/20)/(√3/5)")


# --- B3: Anscombe ---------------------------------------------------------------------------------------------------------------------------------

def test_b3_anscombe_zahlen_und_exakte_beziehungen():
    sets = {satz: X.anscombe(satz) for satz in C.SAETZE}
    for satz, f in sets.items():
        assert sum(f["h"]) == 2
        sst = sum((v - f["yb"]) ** 2 for v in f["y"])
        assert 0.666 <= float(1 - f["sse"] / sst) <= 0.667
        for i in range(f["n"]):
            if f["h"][i] != 1:
                assert X.cook_by_deletion(f, i) == f["D"][i], (satz, i)             # Cook-Formel gleich Löschen und Neuanpassen, Bruch gleich Bruch
    for satz in ("I", "II", "III"):
        f = sets[satz]
        for i in range(11):
            assert f["e"][i] == (1 - f["h"][i]) * X.loeschresiduum(f, i)           # e = (1 − h)·(y − ŷ_(i)) für alle 33 Punkte
    b0s = [float(sets[s]["b0"]) for s in C.SAETZE]
    assert [round(v, 4) for v in b0s] == [3.0001, 3.0009, 3.0025, 3.0017]
    dmax = {s: max(d for d in sets[s]["D"] if d is not None) for s in C.SAETZE}
    _shows(float(dmax["I"]), "0,489", 3)
    _shows(float(dmax["II"]), "0,808", 3)
    _shows(float(dmax["III"]), "1,393", 3)
    _shows(float(dmax["IV"]), "0,137", 3)
    assert sets["II"]["D"][5] == sets["II"]["D"][7]
    f3 = sets["III"]
    _shows(float(f3["h"][2]), "0,236", 3)
    g = X.loo(f3["x"], f3["y"], 2)
    assert (round(float(g["b0"]), 4), round(float(g["b1"]), 4)) == (4.0056, 0.3454)
    f4 = sets["IV"]
    assert f4["h"][7] == 1 and f4["e"][7] == 0 and f4["D"][7] is None and X.loo(f4["x"], f4["y"], 7) is None
    assert all(f4["h"][i] == F(1, 10) for i in range(11) if i != 7)
    assert ev.analyse_anscombe("II", "Größtes Cook-D").quad_r2 > 0.999999 - 5e-7
    _shows(ev.analyse_anscombe("II", "Größtes Cook-D").quad_r2, "0,999999", 6)
    _has("y ≈ 3,00 + 0,50·x", "R² 0,666 bis 0,667", "4/n = 0,364", "F(2, 9; 0,5) = 0,749", "y = 4,0056 + 0,3454·x", "exakt 1/10", "3,0001 bis 3,0025", "alle 33 Punkte",
         "**Hebel genau 1**", "Gleichstand", "Satz IV 0,137 (Punkt 4)")
    assert round(4 / 11, 3) == 0.364 and round(float(stats.f.ppf(0.5, 2, 9)), 3) == 0.749


# --- B4: Zusatzpunkt ------------------------------------------------------------------------------------------------------------------------------

def test_b4_tabellen_stehen_zeilenweise_in_der_readme():
    for tab in (T.tabelle_zusatz(FROZEN), T.tabelle_hebel_anteil(FROZEN), T.tabelle_normal(FROZEN)):
        for zeile in tab.splitlines():
            assert zeile in README, zeile


def test_b4_einzelzahlen_des_textes():
    vz = "Verweilzeit"
    r30, r55, r10, r30_1 = _row(vz, 30.0, 0.5), _row(vz, 5.5, 0.5), _row(vz, 10.0, 0.5), _row(vz, 30.0, 1.0)
    _shows(100 * abs(r30["med_rel"]), "49,0", 1)
    _shows(r30["med_e_s"], "1,63", 2)
    _shows(r55["med_e_s"], "1,65", 2)
    _shows(r30["med_D"], "17,488", 3)
    _shows(r55["med_D"], "0,048", 3)
    assert [ev.de(_row(vz, v, 0.5)["med_e_s"], 2) for v in C.V_ALARM_WERTE] == ["1,65", "2,39", "2,47", "1,63"]            # nicht monoton
    assert [ev.de(_row(vz, v, 0.5)["med_t_ext"], 2) for v in C.V_ALARM_WERTE] == ["1,73", "2,85", "3,61", "4,24"]          # steigt
    _has("(1,65 / 2,39 / 2,47 / 1,63 für Vorlauf 5,5 / 10 / 15 / 30)", "(t* 1,73 / 2,85 / 3,61 / 4,24)")
    _shows(100 * abs(r30_1["med_rel"]), "4,0", 1)
    _shows(r30_1["med_se_ratio"], "0,524", 3)
    assert [ev.pct(_row(vz, v, 0.5)["share_hebel"]) for v in C.V_ALARM_WERTE] == ["0,0 %", "34,7 %", "100,0 %", "100,0 %"]
    _has("bei Vorlauf 5,5 in keinem Datensatz", "bei Vorlauf 10 (Median knapp darunter) in 34,7 %", "bei Vorlauf 15 und 30 in allen")
    _shows(r10["med_h"], "0,120", 3)
    _shows(100 * abs(r10["med_rel"]), "11,2", 1)
    _shows(100 * r10["alarm_D"], "11,8", 1)
    _shows(100 * r10["alarm_T"], "4,0", 1)
    assert max(r10["alarm_D"], r10["alarm_T"]) < 0.12                                                                      # „höchstens 12 %“
    n10 = _row("Normal", 10.0, 0.5)
    _shows(100 * n10["alarm_D"], "44,8", 1)
    _shows(100 * n10["alarm_T"], "15,9", 1)
    _shows(100 * abs(n10["med_rel"]), "10,8", 1)
    _has("trennt Cook zu 45 %, t* zu 16 %", "höchstens 12 %")
    thr = FROZEN["b"][vz]["thr5"]
    _shows(thr["D"], "0,844", 3)
    _shows(thr["T"], "5,65", 2)
    n1 = A.schwellen(31)
    assert (round(n1["cook_4n"], 3), round(n1["cook_f50"], 3)) == (0.129, 0.710)
    _has("4/n = 0,129, F(2, 29; 0,5) = 0,710")


def test_b4_vehikel_zahlen():
    _shows(S.C_MEAN, "1,029", 3)
    _shows(S.SD_POOLED, "1,640", 3)
    assert abs(S.SD_POOLED - 1.638) < 0.005                                                                                # frühere Stichprobenmessung 1,638
    _has("E[d | v] = 1,029·(0,6 + 0,85·v)", "konstante Standardabweichung 1,640")


# --- B5: Falschalarm ------------------------------------------------------------------------------------------------------------------------------

def test_b5_tabelle_und_orakelzahlen():
    for zeile in T.tabelle_falschalarm(FROZEN).splitlines():
        assert zeile in README, zeile
    cn, cv = FROZEN["c"]["Normal"], FROZEN["c"]["Verweilzeit"]
    _shows(100 * cn["cook_4n"], "90,8", 1)
    _shows(100 * cv["cook_4n"], "97,3", 1)
    _shows(cn["mean_count_4n"], "1,73", 2)
    _shows(cv["mean_count_4n"], "2,09", 2)
    _shows(100 * cv["t3"], "56,0", 1)
    _shows(100 * cn["t3"], "17,2", 1)
    o = T.orakel_zahlen()
    _shows(o["beta"], "1,75", 2)
    assert abs(o["beta"] - cn["mean_count_4n"]) < 4 * cn["se_count_4n"]                                                    # Beta-Orakel gegen Messung, 4 Standardfehler
    _shows(o["t_erwartet"], "0,172", 3)
    _shows(o["t_gemessen"], "0,176", 3)
    assert abs(o["t_erwartet"] - o["t_gemessen"]) < 0.02
    _shows(cn["se_count_4n"], "0,013", 3)
    _has("Beta(1/2, (n − p − 1)/2)", "27 Freiheitsgraden", "**nicht getrennt**")


# --- B6: Maskierung -------------------------------------------------------------------------------------------------------------------------------

def test_b6_tabelle_und_einzelzahlen():
    for zeile in T.tabelle_maskierung(FROZEN).splitlines():
        assert zeile in README, zeile
    d = FROZEN["d"]
    _shows(d["einzeln"]["med_D"][0], "17,45", 2)
    _shows(d["paar"]["med_D"][0], "0,68", 2)
    _shows(d["paar"]["med_D"][1], "1,04", 2)
    _shows(100 * d["paar"]["df50_beide"], "15,4", 1)
    _shows(100 * d["einzeln"]["df50_einer"], "98,6", 1)
    _shows(100 * d["paar"]["med_rel"], "−54,1", 1)
    _shows(100 * d["einzeln"]["med_rel"], "−48,7", 1)
    _shows(d["einzeln"]["med_e_s_v2"], "1,65", 2)
    _shows(d["paar"]["med_e_s_v2"], "1,15", 2)
    _shows(abs(d["loeschen_se"]["ohne_beide"]), "4,09", 2)
    assert abs(d["paar"]["med_rel"]) > abs(d["einzeln"]["med_rel"])                                                        # im Paar sinkt die Steigung stärker
    assert d["paar"]["df50_beide"] < d["einzeln"]["df50_einer"] - 0.5


# --- B7: Kunde mit einem Container ---------------------------------------------------------------------------------------------------------------

def test_b7_tabelle_exakt_und_handbeispiel():
    for zeile in T.tabelle_kunde().splitlines():
        assert zeile in README, zeile
    assert [ev.pct(float(X.p_mindestens_ein_einzelner(n))) for n in (12, 24, 36, 60, 120)] == ["99,7 %", "98,5 %", "88,0 %", "31,4 %", "0,4 %"]
    assert X.p_mindestens_ein_einzelner(4, 3) == F(20, 27)
    _has("20/27", "von 81 Zuordnungen haben 21", "(eˣ − x)¹²")
    for n in (36, 60):                                                                                                       # Simulation als Band (4 Standardfehler)
        je = S.container_je_kunde(S.zuordnung(4242, 20000, n))
        p = float(X.p_mindestens_ein_einzelner(n))
        assert abs(float((je == 1).any(axis=1).mean()) - p) < 4 * math.sqrt(p * (1 - p) / 20000)


# --- B8: Stabilität -------------------------------------------------------------------------------------------------------------------------------

def test_b8_stabilitaet_mit_vier_anderen_zufalls_seeds():
    st = Z.stabilitaet(FROZEN)
    spann_anteil, pfad_a = st["anteil"]
    spann_median, pfad_m = st["median"]
    _shows(100 * spann_anteil, "3,3", 1)
    _shows(100 * spann_median, "12,4", 1)
    assert pfad_a == "/b/Normal/rows[7]/alarm_T" and pfad_m == "/d/paar/med_D[1]"
    assert spann_anteil <= 0.04 and spann_median <= 0.15                                                                    # Band der Tests: ±4 Prozentpunkte, ±15 %
    _has("Anteile ±4 Prozentpunkte, Mediane ±15 %")


def test_b8_steigungs_mediane_bei_faktor_1_ueber_fuenf_laeufe():
    """README B8: Steigung bei Faktor 1 über die Datei und vier Zufalls-Seed-Versätze (Welt Verweilzeit, 4000 Datensätze, ohne 5-%-Schwellen)."""
    import hdi_measure as M
    vz = "Verweilzeit"
    werte = {10.0: [], 15.0: [], 30.0: []}
    for versatz in (0, 3000, 7000, 11000, 20000):
        rows = M.messung_b(vz, Z.SEEDS_B[vz] + versatz, C.R_FROZEN_B, C.N_FROZEN, C.V_ALARM_WERTE, C.F_ALARM_WERTE, None)
        for r in rows:
            if r["f"] == 1.0 and r["v"] in werte:
                werte[r["v"]].append(100 * r["med_rel"])
    spannen = {v: (ev.de(max(w), 1), ev.de(min(w), 1)) for v, w in werte.items()}
    assert spannen == {10.0: ("−0,7", "−1,0"), 15.0: ("−1,5", "−2,2"), 30.0: ("−2,5", "−4,0")}
    _has("Vorlauf 10 von −0,7 bis −1,0 %", "Vorlauf 15 von −1,5 bis −2,2 %", "Vorlauf 30 von −2,5 bis −4,0 %")


# --- B9 und Berichtigungen ------------------------------------------------------------------------------------------------------------------------

def test_korrekturen_zahlen_dieser_demo():
    cn, cv = FROZEN["c"]["Normal"], FROZEN["c"]["Verweilzeit"]
    _shows(100 * cn["cook_f50"], "0,4", 1)
    _shows(100 * cv["cook_f50"], "9,7", 1)
    r30 = _row("Verweilzeit", 30.0, 0.5)
    r55 = _row("Verweilzeit", 5.5, 0.5)
    _shows(r30["med_D"], "17,5", 1)
    ratio = r30["med_D"] / r55["med_D"]
    assert 350 < ratio < 375                                                                                                # „etwa 360“
    _has("17,488/0,048", "etwa 360", "früher 91,2 %", "früher 0,5 % und 9,2 %", "früher 17,2 % und 55,7 %", "früher −48,6 %", "42,9 bis 45,8 %", "früher 1,638")
    assert 42.9 <= 100 * _row("Normal", 10.0, 0.5)["alarm_D"] <= 45.8 and 15.1 <= 100 * _row("Normal", 10.0, 0.5)["alarm_T"] <= 17.2     # innerhalb der früheren Spanne


def test_readme_nennt_keine_planungswoerter():
    for wort in ("Stück", "Linie", "Vormessung", "Phase 0", "Plan Z", "geplant", "trägt", "läuft lokal", "Saatwert"):
        assert wort not in README, wort
