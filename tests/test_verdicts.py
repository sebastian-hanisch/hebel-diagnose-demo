"""Verdikte an von Hand gesetzten Werten und an den Presets (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text). Die Zahlen der Handbeispiele und des Quartetts sind in Brüchen von Hand gerechnet (test_handbeispiele.py);
die Verdikte der Zufallsdatensätze nennen ihre Zahlen aus der Rechnung und werden hier gegen die Rechnung gelesen. Ein Preset ohne Effekt meldet keinen Effekt."""

from fractions import Fraction as F

import pytest

import hdi_constants as C
import hdi_evaluation as ev
import hdi_exact as X
import hdi_frozen as Z
import hdi_presets as P


def _settings(name):
    pr = C.PRESETS[name]
    return ev.Settings(**{spec.field: pr[spec.url_param] for spec in P.SETTING_SPECS.values() if spec.url_param in pr})


# --- Von Hand ---------------------------------------------------------------------------------------------------------------------------------

def test_hand_b_y6_gleich_0_nennt_alle_zahlen_von_hand():
    ex, base = ev.analyse_hand(0, "Größtes Cook-D")
    kind, text = ev.verdict_hand(ex.f, base)
    assert kind == "warning"
    for teil in ("von 4/5 auf 1/5", "Sein Residuum ist −1 (Platz 3 der Beträge)", "sein Cook-D 152/35 = 4,34", "höchstens 0,30", "Anteil 5/24 der Abweichung −24/5", "(Hebel 19/24)"):
        assert teil in text, teil


def test_hand_b_auf_der_geraden_meldet_keine_verschiebung():
    ex, base = ev.analyse_hand(24, "Größtes Cook-D")
    kind, text = ev.verdict_hand(ex.f, base)
    assert kind == "success"
    assert "y = 24/5" in text and "Die Gerade bleibt y = 4/5·x, Residuum 0, Cook-D 0" in text and "0,433" in text
    assert "verschiebt" not in text


def test_hand_b_mittlerer_wert_von_hand():
    """y6 = 12/5: Gerade durch die Punkte; von Hand: Steigung 1/2, Residuum −1/2, Cook-D 19/10, Löschresiduum −12/5 (Faktor 5/24)."""
    ex, base = ev.analyse_hand(12, "Größtes Cook-D")
    kind, text = ev.verdict_hand(ex.f, base)
    assert kind == "warning"
    for teil in ("auf 1/2", "Residuum ist −1/2", "Cook-D 19/10 = 1,90", "Abweichung −12/5"):
        assert teil in text, teil
    assert ex.f["D"][5] == F(19, 10) and ex.f["e"][5] == F(-1, 2) and ex.f["b1"] == F(1, 2)


def test_hand_b_art_schwelle_bei_f_median():
    """Die Art wird `warning`, sobald Cook-D des sechsten Punkts über F(2, 4; 0,5) = 2(√2 − 1) = 0,828 liegt; darunter `info`."""
    kinds = {k: ev.verdict_hand(*(lambda r: (r[0].f, r[1]))(ev.analyse_hand(k, "Größtes Cook-D")))[0] for k in (0, 6, 12, 18, 22, 24)}
    assert kinds[0] == "warning" and kinds[24] == "success" and kinds[22] == "info"
    for k in (0, 6, 12, 18, 22):
        d6 = float(ev.analyse_hand(k, "Größtes Cook-D")[0].f["D"][5])
        assert (kinds[k] == "warning") == (d6 > 2 * (2 ** 0.5 - 1)), k


def test_hand_a_verdikt_nennt_hebel_residuen_cook():
    kind, text = ev.verdict_hand_a(X.hand_a())
    assert kind == "info"
    assert "Hebel 1/3, 1/3, 1/3, 1" in text and "Residuen −1, 0, 1, 0" in text and "Cook-D 3/8, 0, 3/8, 0/0" in text and "nicht bestimmt" in text


# --- Anscombe ---------------------------------------------------------------------------------------------------------------------------------

def test_anscombe_iv():
    kind, text = ev.verdict_anscombe("IV", ev.analyse_anscombe("IV", "Größter Hebel"))
    assert kind == "warning"
    for teil in ("Punkt 8 (x = 19) hat Hebel genau 1, Residuum 0", "nicht definiertes Cook-D (0/0)", "ohne ihn ist die Steigung nicht bestimmt", "Größtes Cook-D: 0,137 bei Punkt 4"):
        assert teil in text, teil


def test_anscombe_iii_ohne_punkt_3():
    kind, text = ev.verdict_anscombe("III", ev.analyse_anscombe("III", "Größtes Cook-D"))
    assert kind == "warning"
    for teil in ("Größtes Cook-D: 1,393 bei Punkt 3 (Hebel 0,236)", "Ohne Punkt 3 lautet die Gerade y = 4,0056 + 0,3454·x", "7,6·10⁻⁵"):
        assert teil in text, teil
    assert text.count("darüber") == 3                           # über 4/n, F(.5) und 1


def test_anscombe_ii_gleichstand_und_kurve():
    kind, text = ev.verdict_anscombe("II", ev.analyse_anscombe("II", "Größtes Cook-D"))
    assert kind == "warning" and "Größtes Cook-D: 0,808" in text
    assert "Gleichstand: die Punkte 6 und 8" in text and "eine Kurve" in text and "0,999999" in text


def test_anscombe_i_ueber_vier_durch_n_aber_unter_f_median():
    kind, text = ev.verdict_anscombe("I", ev.analyse_anscombe("I", "Größtes Cook-D"))
    assert kind == "info" and "0,489 bei Punkt 3" in text
    assert "4/n = 0,364 (darüber)" in text and "F(2, 9; 0,5) = 0,749 (darunter)" in text
    assert "Kurve" not in text and "Gleichstand" not in text


def test_anscombe_alle_vier_sagen_dieselbe_gerade():
    for satz in C.SAETZE:
        assert "y = 3,00 + 0,50·x" in ev.verdict_anscombe(satz, ev.analyse_anscombe(satz, "Größtes Cook-D"))[1]


# --- Presets: je Preset die erwartete Verdikt-Zeile ----------------------------------------------------------------------------------------------

def test_preset_sauberer_datensatz_meldet_keinen_zusatzpunkt_effekt():
    s = _settings("Sauberer Datensatz")
    a = ev.analyse_zusatz(s)
    kind, text = ev.verdict_zusatz(a, Z.zeile(Z.laden(), "Verweilzeit", s.v, s.f))
    assert kind == "info" and "ändert der Zusatzpunkt die Steigung nur um" in text and "verschiebt der Zusatzpunkt" not in text
    assert abs(a.rel) < 0.10 and a.d < 4 / (s.n + 1)


def test_preset_halbierter_hebelpunkt_meldet_den_effekt():
    s = _settings("Halbierter Hebelpunkt (Vorlauf 30)")
    a = ev.analyse_zusatz(s)
    kind, text = ev.verdict_zusatz(a, Z.zeile(Z.laden(), "Verweilzeit", s.v, s.f))
    assert kind == "warning" and "verschiebt der Zusatzpunkt die Steigung um" in text and "nach unten" in text
    assert a.rel < -0.4 and a.h > 0.6 and a.d > 1.0
    assert "Vorgerechnet (Median über 4000 Datensätze, n = 30)" in text


def test_preset_aus_demselben_prozess_hat_hohen_hebel_aber_geringe_wirkung():
    s = _settings("Hebelpunkt aus demselben Prozess")
    a = ev.analyse_zusatz(s)
    kind, text = ev.verdict_zusatz(a, Z.zeile(Z.laden(), "Verweilzeit", s.v, s.f))
    assert kind == "info" and a.h > 2 * 2 / (s.n + 1) and abs(a.rel) < 0.10 and a.se_ratio < 0.7
    assert "über 2p/n" in text


def test_zusatz_ohne_vorgerechnete_zeile_nennt_das_raster():
    a = ev.analyse_zusatz(ev.Settings(v=12.5, f=0.75))
    text = ev.verdict_zusatz(a, None)[1]
    assert "keine vorgerechnete Zeile" in text and "5,5, 10,0, 15,0, 30,0" in text


def test_preset_zwei_hebelpunkte_meldet_verdeckung():
    s = _settings("Zwei Hebelpunkte")
    live = ev.analyse_paar(s).live
    kind, text = ev.verdict_paar(s, live)
    assert kind == "warning" and "verdecken sich gegenseitig" in text
    assert live["paar"]["df50_beide"] < live["einzeln"]["df50_einer"] - 0.10


def test_verdikt_paar_grenze_bei_10_prozentpunkten_ist_inklusive():
    def live(einzeln, beide):
        pk = {"N": 32, "med_h": [0.4, 0.4], "med_D": [1.0, 1.0], "med_t_ext": [1.5, 1.5], "df50_einer": 0.9, "df50_beide": beide, "med_rel": -0.5}
        ek = {"N": 31, "med_h": [0.7], "med_D": [10.0], "med_t_ext": [4.0], "df50_einer": einzeln, "df50_beide": einzeln, "med_rel": -0.4}
        return {"R": 100, "n": 30, "v1": 28.0, "v2": 30.0, "einzeln": ek, "paar": pk, "loeschen_se": {"ohne_v2": 0.6, "ohne_v1": 0.5, "ohne_beide": 4.0}}
    s = ev.Settings()
    assert ev.verdict_paar(s, live(0.90, 0.80))[0] == "warning"                   # genau 10 Prozentpunkte: verdeckt
    kind, text = ev.verdict_paar(s, live(0.90, 0.81))
    assert kind == "info" and "verdecken sich die beiden Punkte nicht merklich" in text


def test_verdikt_paar_nennt_zahlen():
    pk = {"N": 32, "med_h": [0.39, 0.46], "med_D": [0.7, 1.04], "med_t_ext": [1.5, 1.6], "df50_einer": 0.92, "df50_beide": 0.154, "med_rel": -0.541}
    ek = {"N": 31, "med_h": [0.76], "med_D": [17.45], "med_t_ext": [4.23], "df50_einer": 0.986, "df50_beide": 0.986, "med_rel": -0.487}
    live = {"R": 4000, "n": 30, "v1": 28.0, "v2": 30.0, "einzeln": ek, "paar": pk, "loeschen_se": {"ohne_v2": 0.62, "ohne_v1": 0.48, "ohne_beide": 4.09}}
    kind, text = ev.verdict_paar(ev.Settings(fp=0.5), live)
    assert kind == "warning"
    for teil in ("Vorlauf 28,0 und 30,0", "Cook-D im Paar 0,70 und 1,04 gegen 17,45", "beide über F(.5) in 15,4 %", "der Punkt allein: 98,6 %", "mindestens einer in 92,0 %",
                 "im Paar um 54,1 %", "beim Punkt allein um 48,7 %", "0,62 bzw. 0,48", "Löschen beider um 4,09", "obwohl die Steigung im Paar stärker sinkt"):
        assert teil in text, teil


def test_alarm_verdikt_einzelne_schwelle_und_5_prozent_zeilen():
    tab = [
        {"gruppe": C.SCHWELLEN[1], "schwelle": C.SCHWELLEN[1], "falsch": 0.973, "treffer": 0.994, "alarm_f1": 0.918},
        {"gruppe": C.SCHWELLEN[2], "schwelle": C.SCHWELLEN[2], "falsch": 0.05, "treffer": 0.987, "alarm_f1": 0.797},
        {"gruppe": C.SCHWELLEN[5], "schwelle": "5-%-Schwelle: größtes Cook-D", "falsch": 0.05, "treffer": 0.118, "alarm_f1": 0.051},
        {"gruppe": C.SCHWELLEN[5], "schwelle": "5-%-Schwelle: größtes |t*|", "falsch": 0.05, "treffer": 0.040, "alarm_f1": 0.051},
    ]
    row = {"med_rel": -0.112}
    kind, text = ev.verdict_alarm(ev.Settings(schwelle=C.SCHWELLEN[1], va=10.0), tab, row)
    assert kind == "warning" and "Falschalarm 97,3 %" in text and "Treffer 99,4 %" in text and "Alarm 91,8 % bei Faktor 1" in text and "um 11,2 %" in text
    kind, text = ev.verdict_alarm(ev.Settings(schwelle=C.SCHWELLEN[2], va=30.0), tab, row)
    assert kind == "info"                                        # Treffer − Falschalarm = 0,937 ≥ 0,5
    kind, text = ev.verdict_alarm(ev.Settings(schwelle=C.SCHWELLEN[5], va=10.0), tab, row)
    assert kind == "warning" and "größte Cook-D in 11,8 %, das größte |t*| in 4,0 %" in text and "Faktor 1 in 5,1 % bzw. 5,1 %" in text


def test_zuordnung_verdikt_schwelle_50_prozent():
    z12 = ev.analyse_zuordnung(ev.Settings(nz=12))
    z120 = ev.analyse_zuordnung(ev.Settings(nz=120))
    assert ev.verdict_zuordnung(z12)[0] == "warning" and "99,7 % (exakt)" in ev.verdict_zuordnung(z12)[1]
    kind, text = ev.verdict_zuordnung(z120)
    assert kind == "info" and "0,4 % (exakt)" in text and "n = 120 Container auf 12 Kunden" in text


def test_preset_kunde_mit_einem_container_n36():
    kind, text = ev.verdict_zuordnung(ev.analyse_zuordnung(_settings("Kunde mit einem Container")))
    assert kind == "warning" and "88,0 % (exakt)" in text and "Hebel ist 1" in text


def test_kernfragen_nennen_die_zahlen_von_hand_und_vorgerechnet():
    ex, base = ev.analyse_hand(0, "Größtes Cook-D")
    assert ev.kernfrage_1(ex, base)[1].startswith("Von Hand B (y₆ = 0): mit allen sechs Punkten ist die Steigung 1/5, ohne Punkt 6")
    assert "um −24/5 von der Geraden ohne ihn ab" in ev.kernfrage_2(ex.f)[1] and "1 − h = 5/24" in ev.kernfrage_2(ex.f)[1]
    ex_iv = ev.analyse_anscombe("IV", "Größter Hebel")
    assert "nicht bestimmt" in ev.kernfrage_1(ex_iv, base)[1]
    text = ev.kernfrage_3(Z.laden(), "Verweilzeit")[1]
    assert "Welt Verweilzeit, n = 30" in text and "Vorlauf 10" in text


def test_formatierung():
    assert ev.de(-1.5, 1) == "−1,5" and ev.pct(0.1234) == "12,3 %" and ev.bruch(F(-24, 5)) == "−24/5" and ev.bruch(F(4, 2)) == "2"
    assert ev.gerade(F(0), F(4, 5)) == "y = 4/5·x" and ev.gerade(F(-1, 5), F(1, 5)) == "y = −1/5 + 1/5·x"
    assert ev.gerade_dez(3.0017, 0.4999) == "y = 3,00 + 0,50·x" and ev.gerade_dez(1.0, -0.5) == "y = 1,00 − 0,50·x"


def test_negative_null_wird_ohne_vorzeichen_geschrieben():
    assert ev.de(-0.0004, 2) == "0,00" and ev.de(-0.004, 2) == "0,00"
    assert ev.pct(-0.00001) == "0,0 %" and ev.de(-0.006, 2) == "−0,01"
