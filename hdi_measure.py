"""Messungen über viele Datensätze (Zähleinheit: ein Datensatz): Zusatzpunkt, Falschalarm in sauberen Datensätzen, zwei Zusatzpunkte.

Dieselben Funktionen rechnen die vorgerechneten Tabellen (`hdi_frozen.py`, R = 4000 bis 20 000) und die Live-Zählungen der App (R aus dem Regler).
Voraussetzungen (stehen auch in der README): Anpassung d = b0 + b1·v mit OLS (Kunde ignoriert), Vorlauf ~ Gleich(1; 10), n Container plus Zusatzpunkt(e)."""

import numpy as np

import hdi_algorithm as A
import hdi_scenario as S


def _anhaengen(x, y, spalten_x, spalten_y):
    return np.concatenate([x] + [c[:, None] for c in spalten_x], axis=1), np.concatenate([y] + [c[:, None] for c in spalten_y], axis=1)


def datensatz_schwellen(welt: str, seed: int, R: int, N: int) -> dict:
    """5-%-Schwellen: das 95-%-Quantil des größten Cook-D bzw. des größten |t*| über N Punkte in R sauberen Datensätzen (kein Zusatzpunkt)."""
    d = S.draws(seed, R, N)
    y = S.werte(welt, d)
    r = A.fit_rows(d.v[:, :N], y[:, :N])
    return {"D": float(np.quantile(A.max_ohne_nan(r.D), 0.95)), "T": float(np.quantile(A.max_ohne_nan(np.abs(r.t_ext)), 0.95))}


def messung_b(welt: str, seed: int, R: int, n: int, vs, fs, thr5: dict | None) -> list[dict]:
    """Ein Zusatzpunkt (Spalte n) bei Vorlauf v* mit Faktor f auf seine Verweilzeit; eine Zeile je (v*, f). `thr5`: 5-%-Schwellen {'D', 'T'} oder None."""
    d = S.draws(seed, R, n)
    y = S.werte(welt, d)
    x0, y0 = d.v[:, :n], y[:, :n]
    base = A.fit_rows(x0, y0)
    N = n + 1
    thr = A.schwellen(N)
    rows = []
    for vs_ in vs:
        extra = S.werte_punkt(welt, d, n, vs_)
        for f in fs:
            x, yy = _anhaengen(x0, y0, [np.full(R, float(vs_))], [f * extra])
            r = A.fit_rows(x, yy)
            fl = A.alarme(r, n, thr)
            s = np.sqrt(r.s2)
            with np.errstate(invalid="ignore"):
                row = {
                    "v": float(vs_), "f": float(f),
                    "med_h": float(np.nanmedian(r.h[:, n])),
                    "med_e_s": float(np.nanmedian(np.abs(r.e[:, n]) / s)),
                    "med_t_int": float(np.nanmedian(np.abs(r.t_int[:, n]))),
                    "med_t_ext": float(np.nanmedian(np.abs(r.t_ext[:, n]))),
                    "med_D": float(np.nanmedian(r.D[:, n])),
                    "share_hebel": float(fl["hebel"].mean()), "share_d4n": float(fl["cook_4n"].mean()),
                    "share_df50": float(fl["cook_f50"].mean()), "share_d1": float(fl["cook_1"].mean()), "share_t3": float(fl["t3"].mean()),
                    "med_db1_se": float(np.nanmedian(np.abs((r.b1 - base.b1) / base.se1))),
                    "med_rel": float(np.nanmedian((r.b1 - base.b1) / base.b1)),
                    "med_se_ratio": float(np.nanmedian(r.se1 / base.se1)),
                }
                if thr5 is not None:
                    row["alarm_D"] = float((A.max_ohne_nan(r.D) > thr5["D"]).mean())
                    row["alarm_T"] = float((A.max_ohne_nan(np.abs(r.t_ext)) > thr5["T"]).mean())
            rows.append(row)
    return rows


def messung_c(welt: str, seed: int, R: int, n: int) -> dict:
    """Falschalarm in sauberen Datensätzen (kein Zusatzpunkt): Anteil der Datensätze, in denen mindestens ein Punkt die Schwelle überschreitet."""
    d = S.draws(seed, R, n)
    y = S.werte(welt, d)
    r = A.fit_rows(d.v[:, :n], y[:, :n])
    thr = A.schwellen(n)
    with np.errstate(invalid="ignore"):
        fl = A.alarme(r, None, thr)
        cnt4 = fl["cook_4n"].sum(axis=1)
        cnt_t = fl["t3"].sum(axis=1)
    return {
        "R": R, "n": n,
        **{k: float(v.any(axis=1).mean()) for k, v in fl.items()},
        "mean_count_4n": float(cnt4.mean()), "share_no_4n": float((cnt4 == 0).mean()),
        "mean_count_t3": float(cnt_t.mean()),
        "median_max_D": float(np.median(A.max_ohne_nan(r.D))),
        "mean_h": float(r.h.mean()), "median_max_h": float(np.median(r.h.max(axis=1))),
        "se_count_4n": float(cnt4.std() / np.sqrt(R)),
    }


def messung_d(seed: int, R: int, n: int, abstand: float, f: float, v2: float = 30.0, welt: str = "Verweilzeit") -> dict:
    """Zwei Zusatzpunkte bei Vorlauf v2 − abstand (Spalte n) und v2 (Spalte n + 1), je mit Faktor f, gegen nur den Punkt bei v2. Zähleinheit: Datensatz."""
    d = S.draws(seed, R, n)
    y = S.werte(welt, d)
    x0, y0 = d.v[:, :n], y[:, :n]
    base = A.fit_rows(x0, y0)
    v1 = v2 - abstand
    e1 = f * S.werte_punkt(welt, d, n, v1)
    e2 = f * S.werte_punkt(welt, d, n + 1, v2)
    c1, c2 = np.full(R, v1), np.full(R, v2)
    single = A.fit_rows(*_anhaengen(x0, y0, [c2], [e2]))
    pair = A.fit_rows(*_anhaengen(x0, y0, [c1, c2], [e1, e2]))
    ohne_v2 = A.fit_rows(*_anhaengen(x0, y0, [c1], [e1]))            # Paar ohne den Punkt bei v2
    out = {"R": R, "n": n, "abstand": float(abstand), "f": float(f), "v1": float(v1), "v2": float(v2)}
    for name, r, cols in (("einzeln", single, [n]), ("paar", pair, [n, n + 1])):
        N = r.n
        thr = A.schwellen(N)
        Dx = r.D[:, cols]
        tx = np.abs(r.t_ext[:, cols])
        with np.errstate(invalid="ignore"):
            out[name] = {
                "N": N,
                "med_h": [float(np.nanmedian(r.h[:, c])) for c in cols],
                "med_D": [float(np.nanmedian(r.D[:, c])) for c in cols],
                "med_t_ext": [float(np.nanmedian(np.abs(r.t_ext[:, c]))) for c in cols],
                "df50_einer": float((Dx > thr["cook_f50"]).any(axis=1).mean()), "df50_beide": float((Dx > thr["cook_f50"]).all(axis=1).mean()),
                "d4n_einer": float((Dx > thr["cook_4n"]).any(axis=1).mean()), "d4n_beide": float((Dx > thr["cook_4n"]).all(axis=1).mean()),
                "d1_einer": float((Dx > 1.0).any(axis=1).mean()), "d1_beide": float((Dx > 1.0).all(axis=1).mean()),
                "t3_einer": float((tx > 3.0).any(axis=1).mean()), "t3_beide": float((tx > 3.0).all(axis=1).mean()),
                "med_db1_se": float(np.nanmedian((r.b1 - base.b1) / base.se1)),
                "med_rel": float(np.nanmedian((r.b1 - base.b1) / base.b1)),
                "med_e_s_v2": float(np.nanmedian(np.abs(r.e[:, cols[-1]]) / np.sqrt(r.s2))),
            }
    with np.errstate(invalid="ignore"):
        out["loeschen_se"] = {
            "ohne_v2": float(np.nanmedian((ohne_v2.b1 - pair.b1) / base.se1)),
            "ohne_v1": float(np.nanmedian((single.b1 - pair.b1) / base.se1)),
            "ohne_beide": float(np.nanmedian((base.b1 - pair.b1) / base.se1)),
        }
    return out
