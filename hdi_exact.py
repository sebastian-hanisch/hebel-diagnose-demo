"""Exakte Bruchrechnung (fractions) für die kleinen Datensätze: Gerade, Hebel, Residuen, Cook-Abstand, Löschen und Neuanpassen.

Konvention: p = 2 (Achsenabschnitt und Steigung); Hebel h_i = 1/n + (x_i − x̄)²/Sxx; Cook-Abstand D_i = e_i² h_i / (p s² (1 − h_i)²) mit s² = SSE/(n − p).
Bei Hebel 1 ist D = 0/0 (undefiniert): der Wert ist `None`, nie inf oder nan. Ist Sxx = 0, ist die Steigung nicht bestimmt: `fit` gibt `None` zurück."""

import math
from fractions import Fraction as F

import hdi_constants as C


def parse(values: str, nenner: int = 1) -> list[F]:
    """Ganzzahlen aus einer Zeichenkette als Brüche (Zahl/nenner)."""
    return [F(int(v), nenner) for v in values.split()]


def fit(x, y):
    """Gerade in Brüchen; `None`, wenn Sxx = 0 (Steigung nicht bestimmt). n > 2 ist nötig, damit s² definiert ist."""
    x = [F(v) for v in x]
    y = [F(v) for v in y]
    n = len(x)
    if n <= C.P:
        raise ValueError("Mit höchstens p = 2 Punkten ist die Streuung s² = SSE/(n − p) nicht definiert.")
    xb = sum(x) / n
    yb = sum(y) / n
    sxx = sum((a - xb) ** 2 for a in x)
    if sxx == 0:
        return None
    b1 = sum((a - xb) * (b - yb) for a, b in zip(x, y)) / sxx
    b0 = yb - b1 * xb
    e = [b - (b0 + b1 * a) for a, b in zip(x, y)]
    h = [F(1, n) + (a - xb) ** 2 / sxx for a in x]
    sse = sum(v * v for v in e)
    s2 = sse / (n - C.P)
    d = [None if (hi == 1 or s2 == 0) else ei ** 2 * hi / (C.P * s2 * (1 - hi) ** 2) for ei, hi in zip(e, h)]
    return dict(n=n, x=x, y=y, xb=xb, yb=yb, sxx=sxx, b0=b0, b1=b1, e=e, h=h, sse=sse, s2=s2, D=d, se1_sq=s2 / sxx)


def loo(x, y, i):
    """Anpassung ohne Punkt i (0-basiert); `None`, wenn danach Sxx = 0 ist."""
    return fit([v for j, v in enumerate(x) if j != i], [v for j, v in enumerate(y) if j != i])


def cook_by_deletion(f, i):
    """Cook-Abstand aus Löschen und Neuanpassen: Σ (ŷ_j − ŷ_j(i))² / (p s²). `None`, wenn die Anpassung ohne Punkt i nicht bestimmt oder s² = 0 ist."""
    g = loo(f["x"], f["y"], i)
    if g is None or f["s2"] == 0:
        return None
    return sum(((f["b0"] + f["b1"] * a) - (g["b0"] + g["b1"] * a)) ** 2 for a in f["x"]) / (C.P * f["s2"])


def loeschresiduum(f, i):
    """y_i − ŷ_(i): Abweichung des Punkts von der Geraden, die ohne ihn angepasst wurde (`None`, wenn diese nicht bestimmt ist)."""
    g = loo(f["x"], f["y"], i)
    if g is None:
        return None
    return f["y"][i] - (g["b0"] + g["b1"] * f["x"][i])


def hand_b(y6_fuenftel: int):
    """Von Hand B: fünf Basispunkte plus ein sechster bei x = 6 mit y6 = Wert/5. Rückgabe (Anpassung aller sechs, Anpassung der Basis)."""
    x = list(C.HAND_B_X) + [C.HAND_B_X6]
    y = list(C.HAND_B_Y) + [F(int(y6_fuenftel), 5)]
    return fit(x, y), fit(C.HAND_B_X, C.HAND_B_Y)


def hand_a():
    return fit(C.HAND_A_X, C.HAND_A_Y)


def anscombe(satz: str):
    xs, ys = C.ANSCOMBE[satz]
    return fit(parse(xs), parse(ys, C.ANSCOMBE_NENNER))


def waehle_punkt(f, regel: str) -> int:
    """Index des Punkts nach der Regel; bei Gleichstand der kleinste Index (der Gleichstand wird in `gleichstand` gemeldet)."""
    if regel == "Größter Hebel":
        werte = f["h"]
    elif regel == "Größtes Cook-D":
        werte = [(-1 if d is None else d) for d in f["D"]]
    elif regel == "Größtes Residuum":
        werte = [abs(v) for v in f["e"]]
    else:
        raise ValueError(f"Unbekannte Regel: {regel}")
    return max(range(len(werte)), key=lambda i: (werte[i], -i))


def gleichstand(f, regel: str) -> list[int]:
    """Alle Punkte, die nach der Regel den größten Wert exakt teilen (Bruch gleich Bruch)."""
    if regel == "Größter Hebel":
        werte = f["h"]
    elif regel == "Größtes Cook-D":
        werte = [(-1 if d is None else d) for d in f["D"]]
    else:
        werte = [abs(v) for v in f["e"]]
    m = max(werte)
    return [i for i, w in enumerate(werte) if w == m]


# --- Kunde mit genau einem Container --------------------------------------------------------------------------------------------------------

def anzahl_ohne_einzelnen(n: int, kunden: int) -> int:
    """Zahl der Zuordnungen von n Containern auf `kunden` Kunden, in denen KEIN Kunde genau einen Container hat (ganzzahlig, exakt).
    f_k(m) = Σ_{b ≠ 1} C(m, b) f_{k−1}(m − b), f_0(0) = 1."""
    f = [1] + [0] * n
    for _ in range(kunden):
        g = [0] * (n + 1)
        for m in range(n + 1):
            g[m] = sum(math.comb(m, b) * f[m - b] for b in range(m + 1) if b != 1)
        f = g
    return f[n]


def p_mindestens_ein_einzelner(n: int, kunden: int = C.K_KUNDEN) -> F:
    """Exakte Wahrscheinlichkeit, dass mindestens ein Kunde genau einen von n Containern hat (Kunden gleichverteilt, unabhängig)."""
    return 1 - F(anzahl_ohne_einzelnen(n, kunden), kunden ** n)
