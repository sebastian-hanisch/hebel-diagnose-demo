"""Vorgerechnete Tabellen (eingefroren als `data/hdi_frozen.json`) und ihre Neuberechnung.

Die Tabellen von Experiment 2 bis 4 brauchen Tausende Datensätze (R = 4000 bis 20 000) und werden deshalb mit `tools/hdi_make_frozen.py` einmal gerechnet und als Datei
mitgeliefert; die App liest nur die Datei und kennzeichnet die Zahlen als „vorgerechnet“. `python tools/hdi_make_frozen.py --check` rechnet die Probezelle (die ersten
R_PROBE Datensätze jeder Tabelle, wegen der Präfixeigenschaft des Zufallsstroms dieselben Datensätze wie in der großen Rechnung) neu und vergleicht sie mit der Datei."""

import json
from pathlib import Path

import hdi_constants as C
import hdi_measure as M

DATA = Path(__file__).resolve().parent / "data" / C.FROZEN_FILE
SEEDS_B = {"Verweilzeit": C.SEED_B_VZ, "Normal": C.SEED_B_NORMAL}
SEEDS_THR = {"Verweilzeit": C.SEED_THR_VZ, "Normal": C.SEED_THR_NORMAL}
SEEDS_C = {"Verweilzeit": C.SEED_C_VZ, "Normal": C.SEED_C_NORMAL}


def rechne(R_b=C.R_FROZEN_B, R_c=C.R_FROZEN_C, R_d=C.R_FROZEN_D, R_thr=C.R_FROZEN_THR, versatz=0) -> dict:
    """Alle Tabellen neu rechnen; `versatz` wird auf alle Zufalls-Seeds addiert (Stabilitätsprüfung mit anderen Zufalls-Seeds, die Datei benutzt versatz = 0)."""
    out = {"meta": {"R_b": R_b, "R_c": R_c, "R_d": R_d, "R_thr": R_thr, "n": C.N_FROZEN, "seeds_b": SEEDS_B, "seeds_thr": SEEDS_THR, "seeds_c": SEEDS_C,
                    "seed_d": C.SEED_D, "v_werte": list(C.V_ALARM_WERTE), "f_werte": list(C.F_ALARM_WERTE)}, "b": {}, "c": {}}
    for welt in C.WELTEN:
        thr = M.datensatz_schwellen(welt, SEEDS_THR[welt] + versatz, R_thr, C.N_FROZEN + 1)
        out["b"][welt] = {"thr5": thr, "rows": M.messung_b(welt, SEEDS_B[welt] + versatz, R_b, C.N_FROZEN, C.V_ALARM_WERTE, C.F_ALARM_WERTE, thr)}
        out["c"][welt] = M.messung_c(welt, SEEDS_C[welt] + versatz, R_c, C.N_FROZEN)
    out["d"] = M.messung_d(C.SEED_D + versatz, R_d, C.N_FROZEN, 2.0, 0.5)
    return out


def probe(frozen: dict) -> dict:
    """Probezelle: die ersten R_PROBE Datensätze jeder Tabelle mit den 5-%-Schwellen der Datei."""
    out = {"R": C.R_PROBE, "b": {}, "c": {}}
    for welt in C.WELTEN:
        out["b"][welt] = M.messung_b(welt, SEEDS_B[welt], C.R_PROBE, C.N_FROZEN, C.V_ALARM_WERTE, C.F_ALARM_WERTE, frozen["b"][welt]["thr5"])
        out["c"][welt] = M.messung_c(welt, SEEDS_C[welt], C.R_PROBE, C.N_FROZEN)
    out["d"] = M.messung_d(C.SEED_D, C.R_PROBE, C.N_FROZEN, 2.0, 0.5)
    return out


def laden() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def zeile(frozen: dict, welt: str, v: float, f: float) -> dict | None:
    """Zeile der Tabelle für Vorlauf v und Faktor f (nur Rasterwerte, nur n = 30), sonst None."""
    for r in frozen["b"][welt]["rows"]:
        if abs(r["v"] - v) < 1e-9 and abs(r["f"] - f) < 1e-9:
            return r
    return None


def _flach(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flach(v, pre + "/" + str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flach(v, pre + f"[{i}]"))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[pre] = d
    return out


def _ist_anteil(pfad: str) -> bool:
    letzter = pfad.split("/")[-1]
    return (letzter.startswith(("share", "alarm")) or letzter in ("hebel", "cook_4n", "cook_f50", "cook_1", "t3") or "_einer" in letzter or "_beide" in letzter)


def stabilitaet(frozen: dict, versaetze=(3000, 7000, 11000, 20000)) -> dict:
    """Dieselbe Rechnung mit anderen Zufalls-Seeds (Versatz auf alle Zufalls-Seeds): Spannweite über die fünf Läufe (Datei und vier Versätze) je Anteil bzw. je Median ab 0,1.
    Rückgabe: größte Spannweite der Anteile (Zahl und Pfad) und größte relative Spannweite (Spannweite / größter Betrag) der Mediane ab 0,1 (Zahl und Pfad)."""
    laeufe = [_flach(frozen)] + [_flach(rechne(versatz=v)) for v in versaetze]
    best_anteil, best_median = (0.0, ""), (0.0, "")
    for pfad in laeufe[0]:
        if pfad.startswith(("/probe", "/meta")):
            continue
        werte = [lauf[pfad] for lauf in laeufe]
        spann = max(werte) - min(werte)
        letzter = pfad.split("/")[-1]
        if _ist_anteil(pfad):
            if spann > best_anteil[0]:
                best_anteil = (spann, pfad)
        elif letzter.startswith("med_") or "loeschen_se" in pfad:
            groesst = max(abs(w) for w in werte)
            if groesst >= 0.1 and spann / groesst > best_median[0]:
                best_median = (spann / groesst, pfad)
    return {"anteil": best_anteil, "median": best_median}
