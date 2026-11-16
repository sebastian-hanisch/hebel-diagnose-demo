"""Auswertungen und Verdikte der fünf Experimente. Verdikte sind reine Funktionen `verdict_*(…) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`.

Zähleinheit überall, wo von Anteilen die Rede ist: ein Datensatz (eine Wiederholung). Medianwerte sind Mediane über die Datensätze der genannten Zahl.
Voraussetzungen der Messungen: einfache Gerade d = b0 + b1·v (OLS, Kunde ignoriert), Export-Container (Vorlauf ~ Gleich(1; 10)), n Container plus Zusatzpunkt(e)."""

import math
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np
from scipy import stats

import hdi_algorithm as A
import hdi_constants as C
import hdi_exact as X
import hdi_frozen as Z
import hdi_measure as M
import hdi_scenario as S

SEED_OFFSET = {2: 2_000_003, 3: 3_000_017, 4: 4_000_037, 5: 5_000_011}   # getrennte Zufallsströme je Experiment (kein Vielfaches des Strom-Schritts)
MINUS = "−"


@dataclass(frozen=True)
class Settings:
    seed: int = C.DEFAULT_SEED
    n: int = C.DEFAULT_N
    r: int = C.DEFAULT_R
    y6: int = C.DEFAULT_Y6
    regel: str = C.DEFAULT_PUNKTREGEL
    satz: str = C.DEFAULT_SATZ
    v: float = C.DEFAULT_V
    f: float = C.DEFAULT_F
    schwelle: str = C.DEFAULT_SCHWELLE
    welt: str = C.DEFAULT_WELT
    va: float = C.DEFAULT_V_ALARM
    abstand: float = C.DEFAULT_ABSTAND
    fp: float = C.DEFAULT_F
    nz: int = C.DEFAULT_NZ


# --- Formatierung ----------------------------------------------------------------------------------------------------------------------------

def de(x, nd=2) -> str:
    """Zahl mit Dezimalkomma und echtem Minuszeichen."""
    text = f"{float(x):.{nd}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")                                  # „−0,00“ gibt es nicht
    return text.replace(".", ",").replace("-", MINUS)


def pct(x, nd=1) -> str:
    """Anteil 0..1 als Prozentzahl mit Komma, mit geschütztem Leerzeichen vor dem Prozentzeichen im Text („12,3 %“)."""
    return de(100.0 * x, nd) + " %"


def bruch(x) -> str:
    """Bruch als „a/b“ oder ganze Zahl, mit echtem Minuszeichen."""
    x = F(x)
    return (str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}").replace("-", MINUS)


def gerade(b0, b1) -> str:
    b0, b1 = F(b0), F(b1)
    if b0 == 0:
        return f"y = {bruch(b1)}·x"
    return f"y = {bruch(b0)} {'+' if b1 >= 0 else MINUS} {bruch(abs(b1))}·x"


def gerade_dez(b0, b1, nd=2) -> str:
    return f"y = {de(b0, nd)} {'+' if float(b1) >= 0 else MINUS} {de(abs(float(b1)), nd)}·x"


# --- Experiment 1: kleine Datensätze exakt ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Exakt:
    f: dict                  # Anpassung in Brüchen (hdi_exact.fit)
    punkt: int               # gewählter Punkt (0-basiert)
    regel: str
    gleich: tuple            # alle Punkte mit demselben größten Wert nach der Regel
    ohne: dict | None        # Anpassung ohne den gewählten Punkt (None: Steigung nicht bestimmt)
    loesch_res: F | None     # y − ŷ_(i)
    rang_res: int            # Rang des Betrags des Residuums (1 = größtes)
    quad_r2: float | None    # R² mit quadratischem Term (nur zur Einordnung)


def analyse_exakt(f: dict, regel: str) -> Exakt:
    i = X.waehle_punkt(f, regel)
    absr = [abs(v) for v in f["e"]]
    rang = 1 + sum(1 for v in absr if v > absr[i])
    x = np.array([float(v) for v in f["x"]])
    y = np.array([float(v) for v in f["y"]])
    quad = None
    if len(set(x)) >= 3:
        res = y - np.polyval(np.polyfit(x, y, 2), x)
        quad = float(1.0 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return Exakt(f, i, regel, tuple(X.gleichstand(f, regel)), X.loo(f["x"], f["y"], i), X.loeschresiduum(f, i), rang, quad)


def analyse_hand(y6: int, regel: str):
    f, base = X.hand_b(y6)
    return analyse_exakt(f, regel), base


def analyse_anscombe(satz: str, regel: str) -> Exakt:
    return analyse_exakt(X.anscombe(satz), regel)


def r_quadrat(f: dict) -> float:
    sst = sum((v - f["yb"]) ** 2 for v in f["y"])
    return float(1 - f["sse"] / sst)


def verdict_hand(f: dict, base: dict) -> tuple[str, str]:
    """Verdikt zu Von Hand B: der sechste Punkt bei x = 6 gegen die Gerade der ersten fünf."""
    y6, e6, h6, d6 = f["y"][5], f["e"][5], f["h"][5], f["D"][5]
    ratio = math.sqrt(f["se1_sq"] / base["se1_sq"])
    if e6 == 0 and f["b1"] == base["b1"]:
        return "success", (f"Der sechste Punkt (x = 6, y = {bruch(y6)}) liegt auf der Geraden der ersten fünf: Die Gerade bleibt {gerade(f['b0'], f['b1'])}, Residuum 0, Cook-D 0. "
                           f"Der Standardfehler der Steigung sinkt auf {de(ratio, 3)} des Wertes ohne diesen Punkt: in diesem Beispiel macht ein Hebelpunkt, der zur Geraden passt, die Gerade genauer.")
    andere = max(f["D"][i] for i in range(5))
    rang = 1 + sum(1 for v in f["e"] if abs(v) > abs(e6))
    dr = X.loeschresiduum(f, 5)
    kind = "warning" if float(d6) > float(stats.f.ppf(0.5, C.P, f["n"] - C.P)) else "info"
    return kind, (f"Der sechste Punkt (x = 6, y = {bruch(y6)}) verschiebt die Steigung von {bruch(base['b1'])} auf {bruch(f['b1'])}. "
                  f"Sein Residuum ist {bruch(e6)} (Platz {rang} der Beträge), sein Cook-D {bruch(d6)} = {de(d6)}, bei den fünf anderen höchstens {de(andere)}. "
                  f"Das Residuum zeigt nur den Anteil {bruch(1 - h6)} der Abweichung {bruch(dr)} von der Geraden ohne den Punkt (Hebel {bruch(h6)}).")


def verdict_anscombe(satz: str, ex: Exakt) -> tuple[str, str]:
    f = ex.f
    n = f["n"]
    f50 = float(stats.f.ppf(0.5, C.P, n - C.P))
    thr_4n = 4.0 / n
    hm = max(f["h"])
    ih = f["h"].index(hm)
    dd = [(-1 if d is None else d) for d in f["D"]]
    dmax = max(dd)
    idd = dd.index(dmax)
    gl = [i for i, d in enumerate(dd) if d == dmax]
    teile = [f"Satz {satz}: Gerade {gerade_dez(f['b0'], f['b1'])}, R² = {de(r_quadrat(f), 3)}; in allen vier Sätzen fast dieselbe Gerade (y ≈ 3,00 + 0,50·x)."]
    kind = "info"
    if hm == 1:
        kind = "warning"
        teile.append(f"Punkt {ih + 1} (x = {bruch(f['x'][ih])}) hat Hebel genau 1, Residuum 0 und ein nicht definiertes Cook-D (0/0); ohne ihn ist die Steigung nicht bestimmt, "
                     f"weil alle anderen x gleich sind. Die Steigung hängt an diesem einen Punkt.")
    teile.append(f"Größtes Cook-D: {de(dmax, 3)} bei Punkt {idd + 1} (Hebel {de(f['h'][idd], 3)}); Schwellen: 4/n = {de(thr_4n, 3)} "
                 f"({'darüber' if dmax > thr_4n else 'darunter'}), F(2, {n - 2}; 0,5) = {de(f50, 3)} ({'darüber' if dmax > f50 else 'darunter'}), 1 ({'darüber' if dmax > 1 else 'darunter'}).")
    if dmax > f50 and hm != 1:
        kind = "warning"
    if len(gl) > 1:
        teile.append("Gleichstand: die Punkte " + " und ".join(str(i + 1) for i in gl) + " haben exakt dasselbe Cook-D.")
    g = X.loo(f["x"], f["y"], idd) if dmax >= 0 else None
    if g is not None and dmax > 1:
        teile.append(f"Ohne Punkt {idd + 1} lautet die Gerade {gerade_dez(g['b0'], g['b1'], 4)} (Residuenquadratsumme {de(float(g['sse']) * 1e5, 1)}·10⁻⁵ statt {de(f['sse'], 2)}).")
    if ex.quad_r2 is not None and ex.quad_r2 > 0.9999 and hm != 1:
        teile.append(f"Dieser Satz ist eine Kurve (mit quadratischem Term R² = {de(ex.quad_r2, 6)}), kein einzelner Punkt.")
    return kind, " ".join(teile)


# --- Experiment 2: ein Zusatzpunkt ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Zusatz:
    s: Settings
    x: np.ndarray
    y: np.ndarray            # n Container, danach der Zusatzpunkt
    mit: A.Rows
    ohne: A.Rows
    h: float
    e_s: float
    t_int: float
    t_ext: float
    d: float
    db0_se: float
    db1_se: float
    rel: float
    se_ratio: float
    kurve_v: np.ndarray
    kurve_h: np.ndarray
    kurve_e_s: np.ndarray
    kurve_d: np.ndarray


KURVE_V = np.arange(C.V_MIN, C.V_MAX + 0.25, 0.5)


def analyse_zusatz(s: Settings) -> Zusatz:
    n = s.n
    d = S.draws(s.seed + SEED_OFFSET[2], 1, n)
    y_all = S.werte("Verweilzeit", d)
    x0, y0 = d.v[:, :n], y_all[:, :n]
    ex = S.werte_punkt("Verweilzeit", d, n, s.v)
    x = np.concatenate([x0, [[s.v]]], axis=1)
    y = np.concatenate([y0, (s.f * ex)[:, None]], axis=1)
    mit, ohne = A.fit_rows(x, y), A.fit_rows(x0, y0)
    sd = math.sqrt(float(mit.s2[0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = float((mit.b1[0] - ohne.b1[0]) / ohne.b1[0])
    # Kurve über den gesetzten Vorlauf (derselbe Kunde und dieselbe Normalzahl des Zusatzpunkts, Faktor f)
    G = len(KURVE_V)
    yk = np.concatenate([np.repeat(y0, G, axis=0), (s.f * S.werte_punkt("Verweilzeit", _wiederhole(d, G), n, KURVE_V))[:, None]], axis=1)
    xk = np.concatenate([np.repeat(x0, G, axis=0), KURVE_V[:, None]], axis=1)
    rk = A.fit_rows(xk, yk)
    return Zusatz(
        s, x[0], y[0], mit, ohne, float(mit.h[0, n]), abs(float(mit.e[0, n])) / sd, abs(float(mit.t_int[0, n])), abs(float(mit.t_ext[0, n])), float(mit.D[0, n]),
        float((mit.b0[0] - ohne.b0[0]) / ohne.se0[0]), float((mit.b1[0] - ohne.b1[0]) / ohne.se1[0]), rel, float(mit.se1[0] / ohne.se1[0]),
        KURVE_V, rk.h[:, n], np.abs(rk.e[:, n]) / np.sqrt(rk.s2), rk.D[:, n],
    )


def _wiederhole(d: S.Draws, G: int) -> S.Draws:
    return S.Draws(d.n, np.repeat(d.v, G, axis=0), np.repeat(d.k, G, axis=0), np.repeat(d.z, G, axis=0))


def verdict_zusatz(a: Zusatz, row: dict | None) -> tuple[str, str]:
    """Verdikt zu Experiment 2: die Zahlen dieses einen Datensatzes, dazu (falls vorgerechnet) die Mediane über 4000 Datensätze."""
    s = a.s
    N = s.n + 1
    thr = A.schwellen(N)
    wirkung = abs(a.rel) >= 0.10
    teile = [(f"In diesem Datensatz (Zufalls-Seed {s.seed}, {s.n} Container plus ein Zusatzpunkt bei Vorlauf {de(s.v, 1)} Tagen, Faktor {de(s.f)}) "
              f"{'verschiebt der Zusatzpunkt die Steigung um ' + pct(abs(a.rel)) + (' nach unten' if a.rel < 0 else ' nach oben') if wirkung else 'ändert der Zusatzpunkt die Steigung nur um ' + pct(abs(a.rel))}. "
              f"Hebel {de(a.h, 3)} ({'über' if a.h > thr['hebel'] else 'unter'} 2p/n = {de(thr['hebel'], 3)}), Residuum {de(a.e_s)}·s, Cook-D {de(a.d, 3)} "
              f"({'über' if a.d > thr['cook_4n'] else 'unter'} 4/n, {'über' if a.d > thr['cook_f50'] else 'unter'} F(.5) = {de(thr['cook_f50'], 3)}), |t*| = {de(a.t_ext)}. "
              f"Der Standardfehler der Steigung ändert sich um den Faktor {de(a.se_ratio, 3)}.")]
    if row is not None:
        teile.append(f"Vorgerechnet (Median über {C.R_FROZEN_B} Datensätze, n = {C.N_FROZEN}): Steigung {de(100 * row['med_rel'], 1)} %, Residuum {de(row['med_e_s'])}·s, "
                     f"Cook-D {de(row['med_D'], 3)}, Standardfehler der Steigung ×{de(row['med_se_ratio'], 3)}.")
    else:
        teile.append(f"Für diese Einstellung gibt es keine vorgerechnete Zeile (Raster: Vorlauf {', '.join(de(v, 1) for v in C.V_ALARM_WERTE)}, Faktor 1 und 0,5, n = {C.N_FROZEN}).")
    kind = "warning" if (a.d > thr["cook_f50"] and wirkung) else "info"
    return kind, " ".join(teile)


# --- Experiment 3: wann schlägt der Alarm an? ----------------------------------------------------------------------------------------------

THR_ZEILEN = (
    (C.SCHWELLEN[0], "hebel", "share_hebel"),
    (C.SCHWELLEN[1], "cook_4n", "share_d4n"),
    (C.SCHWELLEN[2], "cook_f50", "share_df50"),
    (C.SCHWELLEN[3], "cook_1", "share_d1"),
    (C.SCHWELLEN[4], "t3", "share_t3"),
)


@dataclass(frozen=True)
class AlarmLive:
    c: dict
    b: dict          # f -> Zeile
    thr5: dict | None


def analyse_alarm_live(s: Settings) -> AlarmLive:
    seed = s.seed + SEED_OFFSET[3]
    thr5 = None
    if s.n == C.N_FROZEN:
        thr5 = Z.laden()["b"][s.welt]["thr5"]          # 5-%-Schwellen gelten für n = 30 Container plus Zusatzpunkt
    c = M.messung_c(s.welt, seed, s.r, s.n)
    rows = M.messung_b(s.welt, seed, s.r, s.n, [s.va], list(C.F_ALARM_WERTE), thr5)
    return AlarmLive(c, {r["f"]: r for r in rows}, thr5)


def alarm_tabelle(frozen: dict, welt: str, va: float, live: AlarmLive | None) -> list[dict]:
    """Zeilen je Schwelle: Falschalarm (Datensätze ohne Zusatzpunkt, irgendein Punkt über der Schwelle), Treffer (halbierter Zusatzpunkt selbst über der Schwelle),
    Alarm bei Faktor 1 (Zusatzpunkt aus demselben Prozess), vorgerechnet und live. Bei der 5-%-Schwelle zählt der Datensatzalarm (irgendein Punkt)."""
    c = frozen["c"][welt]
    r_half, r_one = Z.zeile(frozen, welt, va, 0.5), Z.zeile(frozen, welt, va, 1.0)
    rows = []
    for name, ckey, bkey in THR_ZEILEN:
        rows.append({"gruppe": name, "schwelle": name, "falsch": c[ckey], "treffer": r_half[bkey], "alarm_f1": r_one[bkey],
                     "falsch_live": None if live is None else live.c[ckey], "treffer_live": None if live is None else live.b[0.5][bkey],
                     "alarm_f1_live": None if live is None else live.b[1.0][bkey]})
    for name, key in (("5-%-Schwelle: größtes Cook-D", "alarm_D"), ("5-%-Schwelle: größtes |t*|", "alarm_T")):
        ok = live is not None and live.thr5 is not None
        rows.append({"gruppe": C.SCHWELLEN[5], "schwelle": name, "falsch": 0.05, "treffer": r_half[key], "alarm_f1": r_one[key],
                     "falsch_live": None, "treffer_live": live.b[0.5][key] if ok else None, "alarm_f1_live": live.b[1.0][key] if ok else None})
    return rows


def verdict_alarm(s: Settings, tab: list[dict], row_half: dict) -> tuple[str, str]:
    sel = [r for r in tab if r["gruppe"] == s.schwelle]
    if s.schwelle == C.SCHWELLEN[5]:
        cook = next(r for r in sel if "Cook" in r["schwelle"])
        tt = next(r for r in sel if "|t*|" in r["schwelle"])
        teile = [f"5-%-Schwelle (so gewählt, dass in {pct(0.05)} der Datensätze ohne Zusatzpunkt Alarm kommt): mit einem halbierten Zusatzpunkt bei Vorlauf {de(s.va, 1)} schlägt das größte Cook-D "
                 f"in {pct(cook['treffer'])}, das größte |t*| in {pct(tt['treffer'])} der Datensätze an; mit einem Zusatzpunkt vom Faktor 1 in {pct(cook['alarm_f1'])} bzw. {pct(tt['alarm_f1'])}."]
    else:
        r = sel[0]
        teile = [f"{r['schwelle']}: Falschalarm {pct(r['falsch'])} der Datensätze ohne Zusatzpunkt (mindestens ein Punkt über der Schwelle, n = {C.N_FROZEN}), "
                 f"Treffer {pct(r['treffer'])} bei halbiertem Zusatzpunkt (Vorlauf {de(s.va, 1)}), Alarm {pct(r['alarm_f1'])} bei Faktor 1."]
    teile.append(f"Welt {s.welt}; vorgerechnet über {C.R_FROZEN_B} Datensätze (Falschalarm über {C.R_FROZEN_C}); die Steigung sinkt dabei im Median um {pct(abs(row_half['med_rel']))}.")
    trennt = min(r["treffer"] - r["falsch"] for r in sel)
    return ("warning" if trennt < 0.5 else "info"), " ".join(teile)


# --- Experiment 4: zwei Hebelpunkte --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Paar:
    live: dict
    x: np.ndarray            # Beispiel (Datensatz 0): n Container, dann die beiden Zusatzpunkte
    y: np.ndarray
    steigungen: dict         # Name -> (b0, b1)
    cook: tuple              # Cook-D der beiden Zusatzpunkte im Paar (Beispiel)
    cook_einzeln: float


def analyse_paar(s: Settings) -> Paar:
    seed = s.seed + SEED_OFFSET[4]
    live = M.messung_d(seed, s.r, s.n, s.abstand, s.fp)
    n = s.n
    d = S.draws(seed, 1, n)
    y0 = S.werte("Verweilzeit", d)[:, :n]
    x0 = d.v[:, :n]
    v1, v2 = live["v1"], live["v2"]
    e1 = s.fp * S.werte_punkt("Verweilzeit", d, n, v1)
    e2 = s.fp * S.werte_punkt("Verweilzeit", d, n + 1, v2)
    x = np.concatenate([x0, [[v1, v2]]], axis=1)[0]
    y = np.concatenate([y0, np.stack([e1, e2], axis=1)], axis=1)[0]
    def sl(idx):
        r = A.fit(x[idx], y[idx])
        return float(r.b0[0]), float(r.b1[0])
    alle = np.arange(n + 2)
    st = {"mit beiden": sl(alle), f"ohne den Punkt bei {de(v2, 1)}": sl(np.delete(alle, n + 1)), f"ohne den Punkt bei {de(v1, 1)}": sl(np.delete(alle, n)), "ohne beide": sl(alle[:n])}
    rp = A.fit(x, y)
    rs = A.fit(np.delete(x, n), np.delete(y, n))
    return Paar(live, x, y, st, (float(rp.D[0, n]), float(rp.D[0, n + 1])), float(rs.D[0, n]))


def verdict_paar(s: Settings, live: dict) -> tuple[str, str]:
    e, p, lo = live["einzeln"], live["paar"], live["loeschen_se"]
    verdeckt = e["df50_einer"] - p["df50_beide"] >= 0.10 - 1e-9          # inklusive Grenze, Rundung der Anteile abgefangen
    kopf = (f"Zwei Zusatzpunkte bei Vorlauf {de(live['v1'], 1)} und {de(live['v2'], 1)} (je Faktor {de(s.fp)}, {live['n']} Container, Median bzw. Anteil über {live['R']} Datensätze): "
            f"Cook-D im Paar {de(p['med_D'][0], 2)} und {de(p['med_D'][1], 2)} gegen {de(e['med_D'][0], 2)} bei dem Punkt allein; "
            f"beide über F(.5) in {pct(p['df50_beide'])} der Datensätze (der Punkt allein: {pct(e['df50_einer'])}), mindestens einer in {pct(p['df50_einer'])}. "
            f"Die Steigung ändert sich im Paar um {pct(abs(p['med_rel']))}, beim Punkt allein um {pct(abs(e['med_rel']))}. "
            f"Löschen eines der beiden Punkte verschiebt die Steigung um {de(abs(lo['ohne_v2']))} bzw. {de(abs(lo['ohne_v1']))} Standardfehler der Basis, Löschen beider um {de(abs(lo['ohne_beide']))}.")
    if verdeckt:
        kleiner = all(p["med_D"][i] < e["med_D"][0] and p["med_t_ext"][i] < e["med_t_ext"][0] for i in (0, 1))
        stuetze = " Cook-D und |t*| beider Punkte sind im Median kleiner als beim Punkt allein" if kleiner else ""
        mehr = ", obwohl die Steigung im Paar stärker sinkt" if abs(p["med_rel"]) > abs(e["med_rel"]) else ""
        return "warning", kopf + " Die beiden Punkte verdecken sich gegenseitig:" + stuetze + mehr + "."
    return "info", kopf + " Bei diesem Abstand verdecken sich die beiden Punkte nicht merklich (beide über F(.5) höchstens 10 Prozentpunkte seltener als der Punkt allein)."


# --- Experiment 5: ein Kunde, ein Container ------------------------------------------------------------------------------------------------

R_SIM_ZUORDNUNG = 2000
N_TABELLE = (12, 24, 36, 60, 120)


@dataclass(frozen=True)
class Zuordnung:
    n: int
    kunden: np.ndarray       # Kunde je Container (ein Beispiel-Datensatz)
    je_kunde: np.ndarray     # Container je Kunde
    hebel: np.ndarray        # 1/n_k je Container
    p_exakt: F
    p_sim: float
    p_fehlt_sim: float       # Anteil der Zuordnungen, in denen mindestens ein Kunde keinen Container hat (nur Simulation)
    einzelne: int            # Zahl der Kunden mit genau einem Container im Beispiel


def analyse_zuordnung(s: Settings) -> Zuordnung:
    seed = s.seed + SEED_OFFSET[5]
    sim = S.zuordnung(seed, R_SIM_ZUORDNUNG, s.nz)
    je = S.container_je_kunde(sim)
    k = sim[0]
    jk = je[0]
    return Zuordnung(s.nz, k, jk, 1.0 / jk[k], X.p_mindestens_ein_einzelner(s.nz), float((je == 1).any(axis=1).mean()), float((je == 0).any(axis=1).mean()), int((jk == 1).sum()))


def verdict_zuordnung(z: Zuordnung) -> tuple[str, str]:
    p = float(z.p_exakt)
    text = (f"n = {z.n} Container auf {C.K_KUNDEN} Kunden: Mit Wahrscheinlichkeit {pct(p)} (exakt) hat mindestens ein Kunde genau einen Container; die Simulation mit {R_SIM_ZUORDNUNG} Zuordnungen "
            f"gibt {pct(z.p_sim)}. In dem gezeigten Beispiel haben {z.einzelne} Kunden genau einen Container: Dessen Hebel ist 1, sein Residuum 0, sein Cook-D nicht definiert, "
            f"und sein Kundeneffekt ist von diesem Container nicht zu trennen.")
    return ("warning" if p > 0.5 else "info"), text


def verdict_hand_a(f: dict) -> tuple[str, str]:
    """Der kleine Fall: drei Container von Kunde A (x = 0), einer von Kunde B (x = 4)."""
    return "info", (f"Von Hand: Hebel {', '.join(bruch(h) for h in f['h'])} (je 1/Zahl der Container des Kunden), Residuen {', '.join(bruch(e) for e in f['e'])}, Cook-D "
                    f"{', '.join('0/0' if d is None else bruch(d) for d in f['D'])}. Ohne den vierten Punkt ist die Steigung nicht bestimmt.")


# --- Kernfragen ------------------------------------------------------------------------------------------------------------------------------

def kernfrage_1(ex_hand: Exakt, base: dict) -> tuple[str, str]:
    """Welcher Punkt entscheidet? Von Hand B: ohne den gewählten Punkt gegen mit ihm."""
    f = ex_hand.f
    i = ex_hand.punkt
    if ex_hand.ohne is None:
        return "warning", f"Ohne Punkt {i + 1} ist die Steigung nicht bestimmt: er allein legt sie fest."
    return "info", (f"Von Hand B (y₆ = {bruch(f['y'][5])}): mit allen sechs Punkten ist die Steigung {bruch(f['b1'])}, ohne Punkt {i + 1} (Regel „{ex_hand.regel}“) "
                    f"{bruch(ex_hand.ohne['b1'])}; ohne den sechsten Punkt {bruch(base['b1'])}.")


def kernfrage_2(f: dict) -> tuple[str, str]:
    """Warum zeigt das Residuum es nicht? e = (1 − h)·(y − ŷ_(i)) für den sechsten Punkt."""
    dr = X.loeschresiduum(f, 5)
    return "info", (f"Der sechste Punkt weicht um {bruch(dr)} von der Geraden ohne ihn ab; sein Residuum ist {bruch(f['e'][5])}, weil e = (1 − h)·{bruch(dr)} "
                    f"mit 1 − h = {bruch(1 - f['h'][5])}: Das Residuum zeigt nur diesen Anteil der Abweichung.")


def kernfrage_3(frozen: dict, welt: str) -> tuple[str, str]:
    c = frozen["c"][welt]
    r = Z.zeile(frozen, welt, 10.0, 0.5)
    return "warning", (f"Welt {welt}, n = {C.N_FROZEN}: Die Faustregel 4/n schlägt in {pct(c['cook_4n'])} der Datensätze ohne Zusatzpunkt an (mindestens ein Punkt darüber), F(.5) in {pct(c['cook_f50'])}. "
                       f"Ein halbierter Zusatzpunkt bei Vorlauf 10 senkt die Steigung im Median um {pct(abs(r['med_rel']))}; bei 5 % Falschalarm schlägt Cook in {pct(r['alarm_D'])}, |t*| in {pct(r['alarm_T'])} der Datensätze an.")


# --- Tabellen für die Anzeige ------------------------------------------------------------------------------------------------------------------

def tabelle_exakt(f: dict, brueche: bool) -> list[dict]:
    """Zeilen je Punkt: x, y, Hebel, Residuum, Cook-D; bei den Handbeispielen als Bruch mit Dezimalwert, sonst dezimal. Undefiniertes Cook-D steht als „undefiniert (0/0)“."""
    def zahl(v, nd):
        return f"{bruch(v)} ({de(v, nd)})" if brueche and F(v).denominator != 1 else (bruch(v) if brueche else de(v, nd))
    rows = []
    for i in range(f["n"]):
        d = f["D"][i]
        rows.append({"Punkt": i + 1, "x": zahl(f["x"][i], 2), "y": zahl(f["y"][i], 2), "Hebel": zahl(f["h"][i], 3), "Residuum": zahl(f["e"][i], 3),
                     "Cook-D": "undefiniert (0/0)" if d is None else zahl(d, 3)})
    return rows


def tabelle_b(frozen: dict, welt: str) -> list[dict]:
    """Ergebnistabelle der vorgerechneten Zeilen (Median bzw. Anteil über die Datensätze); Prozentspalten als Zahl × 100."""
    rows = []
    for r in frozen["b"][welt]["rows"]:
        rows.append({
            "Vorlauf v*": r["v"], "Faktor f": r["f"], "Hebel (Median)": r["med_h"], "|Residuum|/s": r["med_e_s"], "|t*|": r["med_t_ext"], "Cook-D": r["med_D"],
            "D > 4/n (%)": 100 * r["share_d4n"], "D > F(.5) (%)": 100 * r["share_df50"], "D > 1 (%)": 100 * r["share_d1"], "|t*| > 3 (%)": 100 * r["share_t3"],
            "Steigung (%)": 100 * r["med_rel"], "SE-Faktor": r["med_se_ratio"], "Alarm Cook (%)": 100 * r["alarm_D"], "Alarm |t*| (%)": 100 * r["alarm_T"],
        })
    return rows
