"""Erzeugt die Zahlentabellen der README aus `data/hdi_frozen.json` (und der exakten Rechnung für Experiment 5), damit Text und Datei nicht auseinanderlaufen.

Aufruf (aus dem Repo-Ordner): `python tools/hdi_readme_tabellen.py` druckt alle Tabellen als Markdown. `tests/test_claims.py` prüft, dass jede dieser Tabellen Zeile für Zeile
in der README steht."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hdi_constants as C  # noqa: E402
import hdi_evaluation as ev  # noqa: E402
import hdi_exact as X  # noqa: E402
import hdi_frozen as Z  # noqa: E402


def _md(kopf, zeilen):
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(z) + " |" for z in zeilen])


def p1(x):
    return ev.pct(x, 1)


def tabelle_zusatz(frozen, welt="Verweilzeit"):
    """Ein Zusatzpunkt: Mediane und Anteile der Zeilen (Vorlauf v*, Faktor f)."""
    kopf = ["Vorlauf v*", "Faktor f", "Hebel", "Residuum/s", "t* (Betrag)", "Cook-D", "Cook über 4/n", "Cook über F(.5)", "t* über 3", "Steigung", "SE-Faktor", "Alarm Cook", "Alarm t*"]
    zeilen = []
    for r in frozen["b"][welt]["rows"]:
        zeilen.append([ev.de(r["v"], 1), ev.de(r["f"], 1), ev.de(r["med_h"], 3), ev.de(r["med_e_s"], 2), ev.de(r["med_t_ext"], 2), ev.de(r["med_D"], 3), p1(r["share_d4n"]),
                       p1(r["share_df50"]), p1(r["share_t3"]), p1(r["med_rel"]), ev.de(r["med_se_ratio"], 3), p1(r["alarm_D"]), p1(r["alarm_T"])])
    return _md(kopf, zeilen)


def tabelle_hebel_anteil(frozen):
    """Anteil der Datensätze, in denen der Hebel des Zusatzpunkts über 2p/n liegt (hängt nur an x, nicht an Welt oder Faktor)."""
    zeilen = []
    for v in C.V_ALARM_WERTE:
        r = Z.zeile(frozen, "Verweilzeit", v, 0.5)
        zeilen.append([ev.de(v, 1), ev.de(r["med_h"], 3), p1(r["share_hebel"])])
    return _md(["Vorlauf v*", "Hebel (Median)", "Hebel über 2p/n"], zeilen)


def tabelle_normal(frozen):
    """Welt Normal: Zeilen mit Faktor 0,5 bei Vorlauf 10 und 30 und mit Faktor 1 bei Vorlauf 30."""
    kopf = ["Vorlauf v*", "Faktor f", "Steigung", "SE-Faktor", "Alarm Cook", "Alarm t*"]
    zeilen = []
    for v, f in ((10.0, 0.5), (30.0, 0.5), (30.0, 1.0)):
        r = Z.zeile(frozen, "Normal", v, f)
        zeilen.append([ev.de(v, 1), ev.de(f, 1), p1(r["med_rel"]), ev.de(r["med_se_ratio"], 3), p1(r["alarm_D"]), p1(r["alarm_T"])])
    return _md(kopf, zeilen)


def tabelle_falschalarm(frozen):
    """Falschalarm in sauberen Datensätzen (n = 30, kein Zusatzpunkt): Anteil der Datensätze mit mindestens einem Punkt über der Schwelle."""
    kopf = ["Schwelle", "Welt Normal", "Welt Verweilzeit"]
    zeilen = []
    for name, key in (("Hebel über 2p/n", "hebel"), ("Cook über 4/n", "cook_4n"), ("Cook über F(.5)", "cook_f50"), ("Cook über 1", "cook_1"), ("t* (Betrag) über 3", "t3")):
        zeilen.append([name, p1(frozen["c"]["Normal"][key]), p1(frozen["c"]["Verweilzeit"][key])])
    zeilen.append(["Punkte über 4/n je Datensatz (Mittel)", ev.de(frozen["c"]["Normal"]["mean_count_4n"], 2), ev.de(frozen["c"]["Verweilzeit"]["mean_count_4n"], 2)])
    return _md(kopf, zeilen)


def tabelle_maskierung(frozen):
    d = frozen["d"]
    e, p, lo = d["einzeln"], d["paar"], d["loeschen_se"]
    kopf = ["Größe", "Punkt bei 30 allein", "Paar (28 und 30)"]
    zeilen = [
        ["Hebel (Median)", ev.de(e["med_h"][0], 3), f"{ev.de(p['med_h'][0], 3)} und {ev.de(p['med_h'][1], 3)}"],
        ["Cook-D (Median)", ev.de(e["med_D"][0], 2), f"{ev.de(p['med_D'][0], 2)} und {ev.de(p['med_D'][1], 2)}"],
        ["t* (Betrag, Median)", ev.de(e["med_t_ext"][0], 2), f"{ev.de(p['med_t_ext'][0], 2)} und {ev.de(p['med_t_ext'][1], 2)}"],
        ["Cook über F(.5), mindestens ein Sonderpunkt", p1(e["df50_einer"]), p1(p["df50_einer"])],
        ["Cook über F(.5), beide Sonderpunkte", p1(e["df50_beide"]), p1(p["df50_beide"])],
        ["t* über 3, mindestens ein Sonderpunkt", p1(e["t3_einer"]), p1(p["t3_einer"])],
        ["t* über 3, beide Sonderpunkte", p1(e["t3_beide"]), p1(p["t3_beide"])],
        ["Änderung der Steigung (Median)", p1(e["med_rel"]), p1(p["med_rel"])],
        ["Residuum/s des Punkts bei 30 (Median)", ev.de(e["med_e_s_v2"], 2), ev.de(p["med_e_s_v2"], 2)],
    ]
    return _md(kopf, zeilen) + "\n\n" + (f"Löschen im Paar, Änderung der Steigung in Standardfehlern der Basis (Median): ohne den Punkt bei 30 {ev.de(lo['ohne_v2'], 2)}, "
                                         f"ohne den Punkt bei 28 {ev.de(lo['ohne_v1'], 2)}, ohne beide {ev.de(abs(lo['ohne_beide']), 2)}.")


def tabelle_kunde():
    """Exakte Wahrscheinlichkeit, dass mindestens ein Kunde genau einen Container hat (12 Kunden)."""
    zeilen = [[str(n), ev.pct(float(X.p_mindestens_ein_einzelner(n)), 1)] for n in (12, 24, 36, 60, 120)]
    return _md(["Container n", "mindestens ein Kunde mit genau einem Container"], zeilen)


TABELLEN = {
    "zusatz": tabelle_zusatz, "hebel_anteil": tabelle_hebel_anteil, "normal": tabelle_normal, "falschalarm": tabelle_falschalarm, "maskierung": tabelle_maskierung,
}


def alle():
    frozen = Z.laden()
    out = {k: f(frozen) for k, f in TABELLEN.items()}
    out["kunde"] = tabelle_kunde()
    return out


if __name__ == "__main__":
    for name, text in alle().items():
        print(f"### {name}\n{text}\n")


def orakel_zahlen():
    """Beta-Orakel (erwartete Zahl Punkte über 4/n je Datensatz, bedingt auf die Hebel) und t-Orakel (|t*| über 3) gegen die Messung der Tabelle (Welt Normal, n = 30, R = 6000)."""
    import numpy as np
    from scipy import stats

    import hdi_algorithm as A
    import hdi_scenario as S
    n, R = C.N_FROZEN, C.R_FROZEN_C
    d = S.draws(C.SEED_C_NORMAL, R, n)
    r = A.fit_rows(d.v[:, :n], S.werte("Normal", d)[:, :n])
    xthr = 4 / n * 2 * (1 - r.h) / (r.h * (n - 2))
    beta = float(np.where(xthr >= 1.0, 0.0, stats.beta.sf(np.minimum(xthr, 1.0), 0.5, (n - 3) / 2)).sum(axis=1).mean())
    t_erw = float(n * 2 * stats.t.sf(3, n - 3))
    frozen = Z.laden()
    return {"beta": beta, "t_erwartet": t_erw, "t_gemessen": frozen["c"]["Normal"]["mean_count_t3"], "gemessen_4n": frozen["c"]["Normal"]["mean_count_4n"]}
