"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hdi_algorithm as A
import hdi_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _strich(fig, b0, b1, x0, x1, name, farbe, dash="solid"):
    fig.add_trace(go.Scatter(x=[x0, x1], y=[b0 + b1 * x0, b0 + b1 * x1], mode="lines", name=name, line=dict(color=farbe, width=3, dash=dash), hoverinfo="skip"))


def build_exakt_streu(ex, titel: str):
    """Punkte mit der Geraden aller Punkte und der Geraden ohne den gewählten Punkt (falls bestimmt); der gewählte Punkt ist rot."""
    f = ex.f
    x = [float(v) for v in f["x"]]
    y = [float(v) for v in f["y"]]
    i = ex.punkt
    pad = 0.06 * (max(x) - min(x) + 1)
    x0, x1 = min(x) - pad, max(x) + pad
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[v for j, v in enumerate(x) if j != i], y=[v for j, v in enumerate(y) if j != i], mode="markers", name="Punkte",
                             marker=dict(color=C.COLOR_POINT, size=10), hovertemplate="x = %{x:.2f}, y = %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x[i]], y=[y[i]], mode="markers", name=f"gewählter Punkt (P{i + 1})", marker=dict(color=C.COLOR_SPECIAL, size=14, symbol="diamond"),
                             hovertemplate="x = %{x:.2f}, y = %{y:.2f}<extra></extra>"))
    _strich(fig, float(f["b0"]), float(f["b1"]), x0, x1, "Gerade mit allen Punkten", C.COLOR_FIT)
    if ex.ohne is not None:
        _strich(fig, float(ex.ohne["b0"]), float(ex.ohne["b1"]), x0, x1, f"Gerade ohne P{i + 1}", C.COLOR_LOO, "dash")
    fig.update_xaxes(title_text="x", range=[x0, x1])
    fig.update_yaxes(title_text="y")
    fig.update_layout(title=dict(text=titel, font=dict(size=14)))
    return _base(fig)


def build_exakt_balken(ex):
    """Hebel (Marken p/n und 2p/n), Betrag des Residuums und Cook-D (Marken 4/n, F(.5) und 1) je Punkt."""
    f = ex.f
    n = f["n"]
    labels = [f"P{i + 1}" for i in range(n)]
    farben = [C.COLOR_SPECIAL if i == ex.punkt else C.COLOR_BAR for i in range(n)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09, subplot_titles=("Hebel", "Betrag des Residuums", "Cook-D"))
    fig.add_trace(go.Bar(x=labels, y=[float(v) for v in f["h"]], marker_color=farben, showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=C.P / n, line_dash="dot", line_color="gray", annotation_text="p/n", annotation_position="top left", row=1, col=1)
    fig.add_hline(y=2 * C.P / n, line_dash="dash", line_color="gray", annotation_text="2p/n", annotation_position="top left", row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=[abs(float(v)) for v in f["e"]], marker_color=farben, showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra></extra>"), row=2, col=1)
    ds = [0.0 if d is None else float(d) for d in f["D"]]
    fig.add_trace(go.Bar(x=labels, y=ds, marker_color=farben, showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra></extra>"), row=3, col=1)
    f50 = float(A.schwellen(n)["cook_f50"])
    for wert, text in ((4.0 / n, "4/n"), (f50, "F(.5)"), (1.0, "1")):
        fig.add_hline(y=wert, line_dash="dash", line_color="gray", annotation_text=text, annotation_position="top left", row=3, col=1)
    for i, d in enumerate(f["D"]):
        if d is None:
            fig.add_annotation(x=labels[i], y=0, text="undefiniert (0/0)", showarrow=False, yshift=14, font=dict(color=C.COLOR_SPECIAL), row=3, col=1)
    return _base(fig, height=560)


def build_zusatz(a):
    s = a.s
    n = s.n
    x, y = a.x, a.y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[:n], y=y[:n], mode="markers", name="Container", marker=dict(color=C.COLOR_POINT, size=8), hovertemplate="Vorlauf %{x:.1f}: %{y:.2f} Tage<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x[n]], y=[y[n]], mode="markers", name="Zusatzpunkt", marker=dict(color=C.COLOR_SPECIAL, size=14, symbol="diamond"),
                             hovertemplate="Vorlauf %{x:.1f}: %{y:.2f} Tage<extra></extra>"))
    x1 = max(float(x.max()), 10.0) + 1.0
    _strich(fig, float(a.mit.b0[0]), float(a.mit.b1[0]), 0.0, x1, "Gerade mit Zusatzpunkt", C.COLOR_FIT)
    _strich(fig, float(a.ohne.b0[0]), float(a.ohne.b1[0]), 0.0, x1, "Gerade ohne Zusatzpunkt", C.COLOR_LOO, "dash")
    fig.update_xaxes(title_text="Vorlauf in Tagen", range=[0, x1])
    fig.update_yaxes(title_text="Verweilzeit in Tagen")
    return _base(fig)


def build_kurve(a):
    """Hebel und Residuum in Einheiten von s (links) sowie Cook-D (rechts, logarithmisch) über den gesetzten Vorlauf des Zusatzpunkts (derselbe Zufallsstrom, Faktor f)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hebel und |Residuum|/s", "Cook-D (logarithmische Achse)"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=a.kurve_v, y=a.kurve_h, mode="lines", name="Hebel h", line=dict(color=C.COLOR_FIT, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=a.kurve_v, y=a.kurve_e_s, mode="lines", name="|Residuum|/s", line=dict(color=C.COLOR_LOO, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=a.kurve_v, y=np.maximum(a.kurve_d, 1e-6), mode="lines", name="Cook-D", line=dict(color=C.COLOR_SPECIAL, width=3)), row=1, col=2)
    n1 = a.s.n + 1
    for wert, text in ((4.0 / n1, "4/n"), (1.0, "1")):
        fig.add_hline(y=wert, line_dash="dash", line_color="gray", annotation_text=text, annotation_position="top left", row=1, col=2)
    fig.add_vline(x=a.s.v, line_dash="dot", line_color="gray")
    fig.update_xaxes(title_text="gesetzter Vorlauf in Tagen", range=[C.V_MIN, C.V_MAX])
    fig.update_yaxes(type="log", row=1, col=2)
    return _base(fig)


ALARM_FARBEN = {"falsch": "#9aa5b1", "treffer": C.COLOR_SPECIAL, "alarm_f1": C.COLOR_FIT2}


def build_alarm(tab, sel_gruppe: str, va: float):
    """Falschalarm, Treffer (halbierter Zusatzpunkt) und Alarm bei Faktor 1 je Schwelle (vorgerechnet als Balken, live als Rauten); die gewählte Schwelle ist umrandet."""
    namen = [r["schwelle"] for r in tab]
    fig = go.Figure()
    for key, label in (("falsch", "Falschalarm (Datensätze ohne Zusatzpunkt)"), ("treffer", f"Treffer (halbierter Zusatzpunkt, Vorlauf {va:g})"), ("alarm_f1", "Alarm bei Faktor 1 (kein Fehler)")):
        fig.add_trace(go.Bar(x=namen, y=[100 * r[key] for r in tab], name=label, marker_color=ALARM_FARBEN[key],
                             marker_line_width=[3 if r["gruppe"] == sel_gruppe else 0 for r in tab], marker_line_color="black",
                             hovertemplate="%{x}: %{y:.1f} %<extra>vorgerechnet</extra>"))
        live = [(r["schwelle"], 100 * r[key + "_live"]) for r in tab if r[key + "_live"] is not None]
        if live:
            fig.add_trace(go.Scatter(x=[a for a, _ in live], y=[b for _, b in live], mode="markers", name="live", showlegend=(key == "falsch"),
                                     marker=dict(symbol="diamond", size=10, color="black"), hovertemplate="%{x}: %{y:.1f} %<extra>live</extra>"))
    fig.update_yaxes(title_text="Anteil der Datensätze (%)", range=[0, 105])
    fig.update_layout(barmode="group")
    return _base(fig, height=430)


def build_paar(p, s):
    x, y = p.x, p.y
    n = len(x) - 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[:n], y=y[:n], mode="markers", name="Container", marker=dict(color=C.COLOR_POINT, size=8), hovertemplate="Vorlauf %{x:.1f}: %{y:.2f} Tage<extra></extra>"))
    fig.add_trace(go.Scatter(x=x[n:], y=y[n:], mode="markers", name="zwei Zusatzpunkte", marker=dict(color=C.COLOR_SPECIAL, size=14, symbol="diamond"),
                             hovertemplate="Vorlauf %{x:.1f}: %{y:.2f} Tage<extra></extra>"))
    x1 = float(x.max()) + 1.0
    farben = (C.COLOR_FIT, C.COLOR_LOO, C.COLOR_FIT2, "#7a7a7a")
    for (name, (b0, b1)), farbe in zip(p.steigungen.items(), farben):
        _strich(fig, b0, b1, 0.0, x1, f"{name}: Steigung {b1:.3f}".replace(".", ","), farbe, "solid" if name == "mit beiden" else "dash")
    fig.update_xaxes(title_text="Vorlauf in Tagen", range=[0, x1])
    fig.update_yaxes(title_text="Verweilzeit in Tagen")
    return _base(fig)


def build_zuordnung(z):
    """Hebel je Container (1/Zahl der Container des Kunden); Container mit Hebel 1 sind rot."""
    labels = [str(i + 1) for i in range(z.n)]
    farben = [C.COLOR_SPECIAL if abs(h - 1.0) < 1e-12 else C.COLOR_BAR for h in z.hebel]
    fig = go.Figure(go.Bar(x=labels, y=z.hebel, marker_color=farben, hovertemplate="Container %{x}: Hebel %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Container", type="category")
    fig.update_yaxes(title_text="Hebel 1/n_k", range=[0, 1.05])
    return _base(fig, height=320)


def build_zuordnung_kurve(punkte, n_gewaehlt, p_gewaehlt):
    """Exakte Wahrscheinlichkeit, dass mindestens ein Kunde genau einen Container hat, über die Zahl der Container."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[a for a, _ in punkte], y=[100 * b for _, b in punkte], mode="lines+markers", name="exakt", line=dict(color=C.COLOR_FIT2, width=3),
                             hovertemplate="n = %{x}: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[n_gewaehlt], y=[100 * p_gewaehlt], mode="markers", name="gewählt", marker=dict(color=C.COLOR_SPECIAL, size=13, symbol="diamond")))
    fig.update_xaxes(title_text="Zahl der Container n (12 Kunden)")
    fig.update_yaxes(title_text="mindestens ein Kunde mit genau 1 Container (%)", range=[0, 102])
    return _base(fig, height=320)
