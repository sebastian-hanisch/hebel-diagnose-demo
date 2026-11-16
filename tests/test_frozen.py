"""Die vorgerechneten Tabellen (`data/hdi_frozen.json`) passen zur Suite: Probezelle neu gerechnet (tools/hdi_make_frozen.py --check), Form der Datei, Verträglichkeit von Probe und großer Rechnung,
Gleichheit der Live-Zählung mit der Tabelle bei gleicher Datensatzzahl."""

import math
import sys
from pathlib import Path

import pytest

import hdi_constants as C
import hdi_frozen as Z
import hdi_measure as M

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hdi_make_frozen as MK  # noqa: E402


@pytest.fixture(scope="module")
def frozen():
    return Z.laden()


def test_probezelle_passt_zur_datei(frozen):
    """Die ersten 200 Datensätze jeder Tabelle, neu gerechnet, gleich der Probe in der Datei (Zahlen relativ 1e-7, Anteile höchstens 2 Datensätze Abstand)."""
    neu = Z.probe(frozen)
    fehler = MK.vergleiche(neu, frozen["probe"], rel=1e-7, band=2.0 / C.R_PROBE)
    assert not fehler, fehler[:5]


def test_make_frozen_check_hat_exitcode_0():
    assert MK.main(["--check"]) == 0


def test_form_der_datei(frozen):
    m = frozen["meta"]
    assert (m["R_b"], m["R_c"], m["R_d"], m["R_thr"], m["n"]) == (C.R_FROZEN_B, C.R_FROZEN_C, C.R_FROZEN_D, C.R_FROZEN_THR, C.N_FROZEN)
    for welt in C.WELTEN:
        rows = frozen["b"][welt]["rows"]
        assert [(r["v"], r["f"]) for r in rows] == [(v, f) for v in C.V_ALARM_WERTE for f in C.F_ALARM_WERTE]    # 4 mal 2 Zeilen
        assert frozen["c"][welt]["R"] == C.R_FROZEN_C and frozen["c"][welt]["n"] == C.N_FROZEN
        assert frozen["b"][welt]["thr5"]["D"] > 0 and frozen["b"][welt]["thr5"]["T"] > 3
    assert frozen["d"]["R"] == C.R_FROZEN_D and frozen["probe"]["R"] == C.R_PROBE


def test_zeile_findet_rasterwerte_und_sonst_none(frozen):
    assert Z.zeile(frozen, "Verweilzeit", 30.0, 0.5)["v"] == 30.0
    assert Z.zeile(frozen, "Verweilzeit", 12.0, 0.5) is None and Z.zeile(frozen, "Normal", 30.0, 0.75) is None


def test_probe_und_grosse_rechnung_sind_vertraeglich(frozen):
    """Die Probe sind die ersten 200 der 4000 Datensätze: ihre Anteile liegen innerhalb von 4,5 Standardfehlern der Teilstichprobe um die der großen Rechnung."""
    for welt in C.WELTEN:
        for gross, probe in zip(frozen["b"][welt]["rows"], frozen["probe"]["b"][welt]):
            for key in ("share_d4n", "share_df50", "share_d1", "share_t3", "alarm_D", "alarm_T"):
                p = gross[key]
                assert abs(probe[key] - p) <= 4.5 * math.sqrt(max(p * (1 - p), 0.0025) / C.R_PROBE) + 1e-9, (welt, gross["v"], gross["f"], key)


def test_live_zaehlung_mit_gleicher_datensatzzahl_reproduziert_die_tabelle(frozen):
    """Dieselben Funktionen und derselbe Zufallsstrom: die Zeile der Tabelle ist die Live-Zeile bei R = 4000 (Zufalls-Seed der Tabelle)."""
    welt = "Verweilzeit"
    thr = frozen["b"][welt]["thr5"]
    live = M.messung_b(welt, Z.SEEDS_B[welt], C.R_FROZEN_B, C.N_FROZEN, [10.0], [0.5], thr)[0]
    row = Z.zeile(frozen, welt, 10.0, 0.5)
    assert MK.vergleiche(live, row, rel=1e-7, band=2.0 / C.R_FROZEN_B) == []


def test_schwellen_5_prozent_gelten_je_welt_und_liegen_nahe_der_messung(frozen):
    """Die 5-%-Schwellen kommen aus 20 000 sauberen Datensätzen mit 31 Punkten; in einer neuen Stichprobe liegt der Datensatzalarm nahe 5 % (Band ±1,0 Prozentpunkte bei R = 20 000 und Standardfehler 0,15)."""
    for welt in C.WELTEN:
        thr = frozen["b"][welt]["thr5"]
        neu = M.datensatz_schwellen(welt, 99, 20000, C.N_FROZEN + 1)
        assert abs(neu["D"] / thr["D"] - 1) < 0.05 and abs(neu["T"] / thr["T"] - 1) < 0.05


def test_falschalarm_wird_je_welt_mit_den_frozen_werten_gelesen(frozen):
    """Die Tabelle nennt Falschalarm für Welt Normal und Verweilzeit getrennt, mindestens ein Punkt je Datensatz."""
    assert frozen["c"]["Normal"]["cook_4n"] > 0.85 and frozen["c"]["Verweilzeit"]["cook_4n"] > 0.9
    assert frozen["c"]["Verweilzeit"]["t3"] > 2 * frozen["c"]["Normal"]["t3"]
