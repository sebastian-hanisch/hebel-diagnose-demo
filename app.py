"""Hebel und Diagnose – ein Punkt entscheidet: interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein einzelner Punkt kann die Gerade bestimmen, und das Residuum zeigt es nicht. Hebel, Cook-Abstand und studentisierte Residuen am Anscombe-Quartett, an Handbeispielen in
Brüchen und an der Verweilzeit von Export-Containern; Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import hdi_constants as C
import hdi_evaluation as ev
import hdi_exact as X
import hdi_frozen as Z
import hdi_layout as L
import hdi_presets as P
import hdi_visualization as V
import modul_frisch

modul_frisch.aktualisieren(__file__)      # geänderte Hilfsmodule nach einem Update neu laden (die Cloud behält sie sonst bis zum Neustart)

st.set_page_config(page_title="Hebel und Diagnose – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _frozen():
    return Z.laden()


@st.cache_data(show_spinner=False)
def _hand(y6, regel):
    return ev.analyse_hand(y6, regel)


@st.cache_data(show_spinner=False)
def _hand_a():
    return X.hand_a()


@st.cache_data(show_spinner=False)
def _anscombe(satz, regel):
    return ev.analyse_anscombe(satz, regel)


@st.cache_data(show_spinner=False)
def _zusatz(seed, n, v, f):
    return ev.analyse_zusatz(ev.Settings(seed=seed, n=n, v=v, f=f))


@st.cache_data(show_spinner=False)
def _alarm_live(seed, n, r, welt, va):
    return ev.analyse_alarm_live(ev.Settings(seed=seed, n=n, r=r, welt=welt, va=va))


@st.cache_data(show_spinner=False)
def _paar(seed, n, r, abstand, fp):
    return ev.analyse_paar(ev.Settings(seed=seed, n=n, r=r, abstand=abstand, fp=fp))


@st.cache_data(show_spinner=False)
def _zuordnung(seed, nz):
    return ev.analyse_zuordnung(ev.Settings(seed=seed, nz=nz))


@st.cache_data(show_spinner=False)
def _zuordnung_kurve(n_gewaehlt):
    punkte = [(n, float(X.p_mindestens_ein_einzelner(n))) for n in range(12, 121, 12)]
    return punkte


def zeige(verdikt):
    kind, text = verdikt
    getattr(st, kind)(text)


def tabelle(rows, spalten_prozent=()):
    df = pd.DataFrame(rows)
    cfg = {c: st.column_config.NumberColumn(format="%.1f %%") for c in spalten_prozent}
    st.dataframe(df, hide_index=True, width="stretch", column_config=cfg)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in hdi_layout.py).
st.html(L.kopf_css(L.dunkles_design()))
with st.container(key="kopf"):
    st.title("🎯 Hebel und Diagnose – ein Punkt entscheidet")
st.markdown(
    """
Ein einzelner Datenpunkt kann die angepasste **Gerade** bestimmen, und am **Residuum** sieht man es ihm nicht an: Es ist nur der Anteil 1 − h der Abweichung, die man ohne den Punkt
gesehen hätte, wobei **h** der **Hebel** des Punkts ist. Die Demo rechnet das zuerst von Hand in Brüchen (sechs Punkte, Anscombe-Quartett), dann an der Verweilzeit von
Export-Containern: ein Zusatzcontainer mit weitem Vorlauf, die Faustregel **4/n** für den Cook-Abstand und was sie bei sauberen Daten zählt, zwei Hebelpunkte, die sich gegenseitig verdecken,
und Kunden mit nur einem Container. Zähleinheit bei allen Anteilen ist ein **Datensatz** (eine Wiederholung).

Weiter unten: **Kernfragen**, die Experimente 2 bis 5, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Diagnose von Einflusspunkten", expanded=True):
    st.markdown(
        """
1. **Hebel h** misst, wie weit der x-Wert eines Punkts vom Mittel aller x liegt; er hängt nur an x, nicht an y. Alle Hebel zusammen ergeben p (hier 2), im Mittel also p/n.
2. **Residuum e** ist der Abstand des Punkts von der angepassten Geraden. **Studentisiert** heißt: durch s·√(1 − h) geteilt; **intern**, wenn s mit dem Punkt gerechnet wird, **extern**, wenn ohne ihn.
3. **Cook-D** misst, wie weit die Anpassung wandert, wenn man den Punkt löscht (in Einheiten von p·s²). Bei Hebel 1 ist D = 0/0 und nicht definiert.
4. **Warum das Residuum täuscht:** e = (1 − h)·(y − ŷ₍ᵢ₎), wobei ŷ₍ᵢ₎ die Vorhersage der ohne den Punkt angepassten Geraden ist. Je größer der Hebel, desto kleiner der Anteil der Abweichung, der im Residuum ankommt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

P.load_permalink_settings()
P.init_session_state_defaults()

with st.sidebar:
    P.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Zufall und Datensatz**")
    st.number_input("Zufalls-Seed", *P.bounds("seed_input"), key="seed_input", step=1, help="Bestimmt die Zufallsdatensätze der Experimente 2 bis 5.")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=P.randomize_seed)
    st.slider("Container im Datensatz n", *P.bounds("n_container"), key="n_container", help="Zahl der Export-Container ohne Zusatzpunkt (Experimente 2 bis 4). Die vorgerechneten Tabellen gelten für n = 30.")
    st.slider("Datensätze der Live-Zählung R", *P.bounds("r_live"), key="r_live", step=20, help="Wie viele Zufallsdatensätze die Experimente 3 und 4 live zählen (die vorgerechneten Tabellen nehmen 4000 bis 20 000).")

s = P.current_settings()
frozen = _frozen()
P.sync_query_params({k: st.session_state[k] for k in P.SETTING_SPECS})

# --- Ergebnis: Experiment 1 ----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: Welcher Punkt entscheidet? (Experiment 1: Von Hand und Anscombe)")
st.radio("Gegenprobe: die Gerade ohne den Punkt mit …", C.PUNKTREGELN, key="punktregel", horizontal=True,
         help="Welcher Punkt herausgenommen wird, um die Gerade ohne ihn zu zeichnen: größter Hebel, größtes Cook-D oder größtes Residuum.")

st.markdown("### 1a – Von Hand: fünf Punkte, ein sechster Punkt bei x = 6")
st.slider("y₆ des sechsten Punkts in Fünfteln (24 = genau auf der Geraden der ersten fünf, 0 = 24/5 darunter)", *P.bounds("y6_fuenftel"), key="y6_fuenftel",
          help="y₆ = Wert/5. Die ersten fünf Punkte (x = −2 bis 2) liegen um die Gerade y = 4x/5.")
ex_a, base = _hand(s.y6, s.regel)
fa = ex_a.f
m1, m2, m3, m4 = st.columns(4)
m1.metric("Steigung der ersten fünf", ev.bruch(base["b1"]))
m2.metric("Steigung mit dem sechsten Punkt", ev.bruch(fa["b1"]))
m3.metric("Hebel des sechsten Punkts", ev.bruch(fa["h"][5]), help="1/6 + (6 − 1)²/40 = 19/24: der sechste Punkt liegt weit rechts.")
m4.metric("Cook-D des sechsten Punkts", ev.bruch(fa["D"][5]) + f" = {ev.de(fa['D'][5])}")
c1, c2 = st.columns(2)
c1.plotly_chart(V.build_exakt_streu(ex_a, "Von Hand: sechs Punkte"), width="stretch", key="chart_hand_streu")
c2.plotly_chart(V.build_exakt_balken(ex_a), width="stretch", key="chart_hand_balken")
tabelle(ev.tabelle_exakt(fa, True))
zeige(ev.verdict_hand(fa, base))
st.caption("Gewählter Punkt: " + f"P{ex_a.punkt + 1}" + (f" (Gleichstand: {', '.join('P' + str(i + 1) for i in ex_a.gleich)})" if len(ex_a.gleich) > 1 else "")
           + " nach der Regel „" + s.regel + "“. Hebel, Residuum und Cook-D sind exakt in Brüchen gerechnet.")

st.markdown("### 1b – Anscombe-Quartett: vier Datensätze, dieselbe Gerade")
st.radio("Anscombe-Satz", C.SAETZE, key="anscombe_satz", horizontal=True,
         help="Vier Datensätze zu je elf Punkten (Anscombe 1973) mit fast gleicher Geraden; Satz IV hat einen Punkt mit Hebel genau 1.")
ex_b = _anscombe(s.satz, s.regel)
d1, d2 = st.columns(2)
d1.plotly_chart(V.build_exakt_streu(ex_b, f"Anscombe Satz {s.satz}"), width="stretch", key="chart_anscombe_streu")
d2.plotly_chart(V.build_exakt_balken(ex_b), width="stretch", key="chart_anscombe_balken")
tabelle(ev.tabelle_exakt(ex_b.f, False))
zeige(ev.verdict_anscombe(s.satz, ex_b))
st.caption(f"Gewählter Punkt: P{ex_b.punkt + 1}" + (f" (Gleichstand: {', '.join('P' + str(i + 1) for i in ex_b.gleich)})" if len(ex_b.gleich) > 1 else "")
           + f" nach der Regel „{s.regel}“. Quelle der Zahlen: {C.ANSCOMBE_QUELLE}. Die Cook-Werte stehen nicht bei Anscombe, sie sind eigene Rechnung.")

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.markdown("## ❓ Kernfragen")
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown("**1. Welcher Punkt entscheidet?**")
    zeige(ev.kernfrage_1(ex_a, base))
with k2:
    st.markdown("**2. Warum zeigt das Residuum es nicht?**")
    zeige(ev.kernfrage_2(fa))
with k3:
    st.markdown("**3. Wann schlägt eine Kennzahl an, und wann zu Unrecht?**")
    zeige(ev.kernfrage_3(frozen, s.welt))

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Ein Zusatzcontainer mit weitem Vorlauf")
st.caption(f"Export-Container (Vorlauf 1 bis 10 Tage, Verweilzeit steigt mit dem Vorlauf), angepasst wird d = b0 + b1·Vorlauf. Zum Datensatz kommt ein Zusatzcontainer bei gesetztem Vorlauf v* "
           f"(über 10 Tage heißt: Schiff verschoben, das ist gesetzt und kommt im Containerstrom nicht vor) mit dem Faktor f auf seine Verweilzeit: f = 0,5 halbiert sie, f = 1 lässt sie, "
           f"wie der Prozess sie liefert. Das gezeigte Beispiel ist ein einzelner Datensatz (Zufalls-Seed {s.seed}); die Tabelle darunter ist vorgerechnet.")
e1, e2 = st.columns(2)
e1.slider("Vorlauf des Zusatzpunkts v* (Tage)", *P.bounds("v_stern"), key="v_stern", step=0.5, format="%.1f")
e2.slider("Faktor f auf die Verweilzeit des Zusatzpunkts", *P.bounds("f_faktor"), key="f_faktor", step=0.25, format="%.2f")
za = _zusatz(s.seed, s.n, s.v, s.f)
q1, q2, q3, q4 = st.columns(4)
q1.metric("Hebel h", ev.de(za.h, 3))
q2.metric("|Residuum| / s", ev.de(za.e_s))
q3.metric("Cook-D", ev.de(za.d, 3))
q4.metric("Steigung", ev.pct(za.rel), help="Änderung der Steigung durch den Zusatzpunkt, in Prozent der Steigung ohne ihn.")
st.caption(f"|t| (intern) = {ev.de(za.t_int)}, |t*| (extern) = {ev.de(za.t_ext)}; Achsenabschnitt verschiebt sich um {ev.de(abs(za.db0_se))}, Steigung um {ev.de(abs(za.db1_se))} Standardfehler der Anpassung ohne Zusatzpunkt; "
           f"Standardfehler der Steigung ×{ev.de(za.se_ratio, 3)}.")
f1, f2 = st.columns(2)
f1.plotly_chart(V.build_zusatz(za), width="stretch", key="chart_zusatz")
f2.plotly_chart(V.build_kurve(za), width="stretch", key="chart_kurve")
row = Z.zeile(frozen, "Verweilzeit", s.v, s.f)
zeige(ev.verdict_zusatz(za, row if s.n == C.N_FROZEN else None))
st.markdown(f"**Vorgerechnet: Welt „Verweilzeit“, n = {C.N_FROZEN} Container plus Zusatzpunkt, Median bzw. Anteil über {C.R_FROZEN_B} Datensätze** (Zufalls-Seed der Rechnung {C.SEED_B_VZ}). "
            f"Anteile: Datensätze, in denen der Zusatzpunkt die Schwelle überschreitet. „Alarm“: irgendein Punkt über der Schwelle, die in sauberen Datensätzen 5 % Falschalarm ergibt "
            f"(größtes Cook-D über {ev.de(frozen['b']['Verweilzeit']['thr5']['D'], 3)}, größtes |t*| über {ev.de(frozen['b']['Verweilzeit']['thr5']['T'], 2)}).")
tabelle(ev.tabelle_b(frozen, "Verweilzeit"), ("D > 4/n (%)", "D > F(.5) (%)", "D > 1 (%)", "|t*| > 3 (%)", "Steigung (%)", "Alarm Cook (%)", "Alarm |t*| (%)"))

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wann schlägt der Alarm an, und wann zu Unrecht?")
st.caption("Falschalarm: Anteil der Datensätze **ohne** Zusatzpunkt, in denen mindestens ein Punkt die Schwelle überschreitet. Treffer: Anteil der Datensätze mit halbiertem Zusatzpunkt, in denen der Zusatzpunkt "
           "selbst die Schwelle überschreitet. Alarm bei Faktor 1: dasselbe für einen Zusatzpunkt, der aus demselben Prozess stammt (kein Fehler). Bei der 5-%-Schwelle zählt der Datensatzalarm "
           "(irgendein Punkt über der Schwelle). „Normal“ heißt: gleiches Mittel, konstante Streuung, normale Fehler; „Verweilzeit“: lognormal, Streuung wächst mit dem Vorlauf.")
g1, g2, g3 = st.columns([2, 1, 2])
g1.selectbox("Schwelle", C.SCHWELLEN, key="schwelle")
g2.radio("Welt", C.WELTEN, key="welt", horizontal=True)
g3.radio("Vorlauf des Zusatzpunkts v*", C.V_ALARM_WERTE, key="v_alarm", horizontal=True, format_func=lambda v: f"{v:g}".replace(".", ","))
with st.spinner(f"Zähle {s.r} Datensätze live …"):
    live3 = _alarm_live(s.seed, s.n, s.r, s.welt, s.va)
tab3 = ev.alarm_tabelle(frozen, s.welt, s.va, live3)
st.plotly_chart(V.build_alarm(tab3, s.schwelle, s.va), width="stretch", key="chart_alarm")
tabelle([{"Schwelle": r["schwelle"], "Falschalarm (vorgerechnet)": 100 * r["falsch"], "Treffer (vorgerechnet)": 100 * r["treffer"], "Alarm bei Faktor 1 (vorgerechnet)": 100 * r["alarm_f1"],
          "Falschalarm (live)": None if r["falsch_live"] is None else 100 * r["falsch_live"], "Treffer (live)": None if r["treffer_live"] is None else 100 * r["treffer_live"],
          "Alarm bei Faktor 1 (live)": None if r["alarm_f1_live"] is None else 100 * r["alarm_f1_live"]} for r in tab3],
        ("Falschalarm (vorgerechnet)", "Treffer (vorgerechnet)", "Alarm bei Faktor 1 (vorgerechnet)", "Falschalarm (live)", "Treffer (live)", "Alarm bei Faktor 1 (live)"))
st.caption(f"Vorgerechnet: Falschalarm über {C.R_FROZEN_C}, Treffer über {C.R_FROZEN_B} Datensätze, n = {C.N_FROZEN}. Live: {s.r} Datensätze mit n = {s.n} (Zufalls-Seed {s.seed}); die 5-%-Schwellen gibt es live nur für n = {C.N_FROZEN}.")
zeige(ev.verdict_alarm(s, tab3, Z.zeile(frozen, s.welt, s.va, 0.5)))

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Zwei Hebelpunkte verdecken sich")
st.caption("Zwei halbierte Zusatzcontainer: einer bei Vorlauf 30 (gesetzt), der andere um den Abstand davor. Verglichen wird mit dem Punkt bei 30 allein. Zähleinheit: ein Datensatz; "
           "Mediane und Anteile über die Datensätze der Live-Zählung.")
h1, h2 = st.columns(2)
h1.slider("Abstand der beiden Zusatzpunkte (Tage)", *P.bounds("abstand"), key="abstand", step=0.5, format="%.1f")
h2.slider("Faktor f auf die Verweilzeit beider Punkte", *P.bounds("f_paar"), key="f_paar", step=0.25, format="%.2f")
with st.spinner(f"Zähle {s.r} Datensätze live …"):
    pa = _paar(s.seed, s.n, s.r, s.abstand, s.fp)
st.plotly_chart(V.build_paar(pa, s), width="stretch", key="chart_paar")
lv = pa.live
tabelle([
    {"Anpassung": "Punkt bei 30 allein", "Hebel (Median)": lv["einzeln"]["med_h"][0], "Cook-D (Median)": lv["einzeln"]["med_D"][0], "|t*| (Median)": lv["einzeln"]["med_t_ext"][0],
     "D > F(.5), mindestens einer (%)": 100 * lv["einzeln"]["df50_einer"], "D > F(.5), beide (%)": 100 * lv["einzeln"]["df50_beide"], "Steigung (%)": 100 * lv["einzeln"]["med_rel"]},
    {"Anpassung": "Paar, Punkt bei Vorlauf " + ev.de(lv["v1"], 1), "Hebel (Median)": lv["paar"]["med_h"][0], "Cook-D (Median)": lv["paar"]["med_D"][0], "|t*| (Median)": lv["paar"]["med_t_ext"][0],
     "D > F(.5), mindestens einer (%)": 100 * lv["paar"]["df50_einer"], "D > F(.5), beide (%)": 100 * lv["paar"]["df50_beide"], "Steigung (%)": 100 * lv["paar"]["med_rel"]},
    {"Anpassung": "Paar, Punkt bei Vorlauf " + ev.de(lv["v2"], 1), "Hebel (Median)": lv["paar"]["med_h"][1], "Cook-D (Median)": lv["paar"]["med_D"][1], "|t*| (Median)": lv["paar"]["med_t_ext"][1],
     "D > F(.5), mindestens einer (%)": 100 * lv["paar"]["df50_einer"], "D > F(.5), beide (%)": 100 * lv["paar"]["df50_beide"], "Steigung (%)": 100 * lv["paar"]["med_rel"]},
], ("D > F(.5), mindestens einer (%)", "D > F(.5), beide (%)", "Steigung (%)"))
st.caption(f"Dieser Datensatz (Zufalls-Seed {s.seed}): Cook-D im Paar {ev.de(pa.cook[0], 2)} und {ev.de(pa.cook[1], 2)}, beim Punkt bei {ev.de(lv['v2'], 1)} allein {ev.de(pa.cook_einzeln, 2)}.")
zeige(ev.verdict_paar(s, pa.live))
d_ = frozen["d"]
st.caption(f"Vorgerechnet (Abstand 2, Faktor 0,5, n = {C.N_FROZEN}, {C.R_FROZEN_D} Datensätze): Cook-D im Paar {ev.de(d_['paar']['med_D'][0], 2)} und {ev.de(d_['paar']['med_D'][1], 2)} gegen {ev.de(d_['einzeln']['med_D'][0], 2)} allein; "
           f"beide über F(.5) in {ev.pct(d_['paar']['df50_beide'])} gegen {ev.pct(d_['einzeln']['df50_einer'])}; Löschen beider verschiebt die Steigung um {ev.de(abs(d_['loeschen_se']['ohne_beide']))} Standardfehler.")

st.markdown("---")

# --- Experiment 5 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Ein Kunde, ein Container")
st.caption("Im Faktor-Modell „Verweilzeit ~ Kunde“ (ein Mittelwert je Kunde, kein Achsenabschnitt) hat ein Container den Hebel 1/n_k, wobei n_k die Zahl der Container seines Kunden ist. "
           "Hat ein Kunde nur einen Container, ist dessen Hebel genau 1: das Modell geht durch diesen Punkt. Die Container verteilen sich gleichmäßig zufällig auf zwölf Kunden.")
fh = _hand_a()
zeige(ev.verdict_hand_a(fh))
st.caption("Handbeispiel: drei Container von Kunde A (x = 0 steht für Kunde A, x = 4 für Kunde B) mit Verweilzeiten 1, 2, 3 und ein Container von Kunde B mit 6; die Gerade ist y = 2 + x.")
st.slider("Container insgesamt n", *P.bounds("n_zuordnung"), key="n_zuordnung", help="Zahl der Container, die auf zwölf Kunden verteilt werden.")
zo = _zuordnung(s.seed, s.nz)
st.plotly_chart(V.build_zuordnung(zo), width="stretch", key="chart_zuordnung")
st.plotly_chart(V.build_zuordnung_kurve(_zuordnung_kurve(s.nz), s.nz, float(zo.p_exakt)), width="stretch", key="chart_zuordnung_kurve")
zeige(ev.verdict_zuordnung(zo))
st.caption(f"Exakt gerechnet über alle Zuordnungen (erzeugende Funktion, ganzzahlig); Simulation: {ev.R_SIM_ZUORDNUNG} Zuordnungen, Zufalls-Seed {s.seed}. In der Simulation fehlt bei n = {s.nz} "
           f"mindestens ein Kunde ganz in {ev.pct(zo.p_fehlt_sim)} der Zuordnungen (nur Simulation, nicht exakt gerechnet).")

st.markdown("---")

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------------

cvz, cno = frozen["c"]["Verweilzeit"], frozen["c"]["Normal"]
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Einfache Gerade (p = 2), n = {C.N_FROZEN} (Quartett n = 11)** | Mehrere Merkmale und Kollinearität sind nicht gemessen; der Hebel ist dort der Abstand zum Schwerpunkt im Merkmalsraum. | [kollinearitaet-auswahl-demo](https://github.com/sebastian-hanisch/kollinearitaet-auswahl-demo) |
| **Gleiche Streuung der Fehler** | Bei der Verweilzeit (Streuung wächst, lognormal) schlägt der Betrag von t* über 3 in {ev.pct(cvz['t3'])} der sauberen Datensätze an, bei normalen Fehlern mit konstanter Streuung in {ev.pct(cno['t3'])}; ob Streuung oder Schiefe schuld ist, ist nicht getrennt. | [streuung-wls-demo](https://github.com/sebastian-hanisch/streuung-wls-demo) |
| **Ein einzelner Zusatzpunkt mit gesetztem Vorlauf** | Vorlauf über 10 Tage kommt im Containerstrom nicht vor; ein Raster über f und n, Zusatzpunkte über dem Prozess (f > 1) und größere Gruppen von Hebelpunkten sind nicht gemessen. | – |
| **Einen Punkt zu löschen ist harmlos** | Löschen nach Ansicht der Daten ist eine Modellwahl; Tests und Intervalle danach gelten nicht mehr wie vorher. Die Demo hat deshalb keinen Löschen-Knopf und keine Empfehlung. | [modellwahl-demo](https://github.com/sebastian-hanisch/modellwahl-demo) |
| **Kunden als Faktor mit mehreren Containern je Kunde** | Mit nur einem Container ist der Hebel 1 und der Kundeneffekt nicht zu trennen; zufällige Kundeneffekte statt fester sind nicht gemessen. | [cluster-gemischt-demo](https://github.com/sebastian-hanisch/cluster-gemischt-demo), [faktoren-codierung-demo](https://github.com/sebastian-hanisch/faktoren-codierung-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben** sind Punkte $(x_i, y_i)$, $i = 1, \dots, n$, und die Gerade $y = b_0 + b_1 x$ nach kleinsten Quadraten; **gesucht** sind die Kennzahlen, die zeigen, welcher Punkt die Anpassung bestimmt.

| Symbol | Bedeutung |
|---|---|
| $n$ | Zahl der Punkte |
| $p$ | Zahl der Spalten der Entwurfsmatrix $X$ einschließlich Achsenabschnitt (hier $p = 2$) |
| $h_i$ | Hebel: $i$-te Diagonale der Hat-Matrix $H = X(X^\top X)^{-1}X^\top$ |
| $e_i$ | Residuum $y_i - \hat y_i$ |
| $s^2$ | $\mathrm{SSE}/(n - p)$ |
| $t_i$, $t_i^*$ | intern bzw. extern studentisiertes Residuum |
| $D_i$ | Cook-Abstand |

**Hebel.** $h_i = \dfrac{1}{n} + \dfrac{(x_i - \bar x)^2}{S_{xx}}$, $\sum_i h_i = p$, also im Mittel $p/n$. Der Hebel hängt nur an $x$.

**Residuum und Hebel.** $e_i = (1 - h_i)\,(y_i - \hat y_{(i)})$ mit $\hat y_{(i)}$ der Vorhersage der ohne Punkt $i$ angepassten Geraden: das Residuum ist nur der Anteil $1 - h_i$ der Abweichung.

**Studentisiert.** $t_i = \dfrac{e_i}{s\sqrt{1 - h_i}}$ (intern), $t_i^* = t_i\sqrt{\dfrac{n - p - 1}{n - p - t_i^2}}$ (extern, $s$ ohne Punkt $i$ gerechnet).

**Cook-Abstand.** $D_i = \dfrac{e_i^2\, h_i}{p\, s^2\,(1 - h_i)^2} = \dfrac{t_i^2}{p}\cdot\dfrac{h_i}{1 - h_i}$ (Cook 1977, Gl. 8); bei $h_i = 1$ ist $D_i = 0/0$ undefiniert. Dasselbe $D_i$ ergibt sich,
wenn man den Punkt löscht und neu anpasst: $D_i = \sum_j (\hat y_j - \hat y_{j(i)})^2/(p s^2)$.

**Konventionen und Schwellen** (eine Auswahl, keine Wahrheit):

| Größe | Wert | Anmerkung |
|---|---|---|
| Mittel der Hebel | $p/n$ | ISLR2 schreibt $(p+1)/n$ mit $p$ = Zahl der Merkmale ohne Achsenabschnitt: dieselbe Zahl, andere Zählung |
| Hebel über | $2p/n$ | bei $p = 2$ dieselbe Zahl wie $4/n$ |
| Cook über | $4/n$ | verbreitete Faustregel; steht nicht bei Cook 1977 |
| Cook über | $F(p, n - p; 0{,}5)$ | Cook 1977: Löschen verschiebt die Schätzung an den Rand des 50-%-Vertrauensellipsoids |
| Cook über | $1$ | |
| $\lvert t^*\rvert$ über | $3$ | |
| Hebel gleich 1 | $D$ undefiniert | die Anpassung geht durch den Punkt, $e = 0$ |

**Faktor-Modell.** Mit einem Kunden als Gruppe (ein Mittelwert je Kunde) ist $h_i = 1/n_k$, $n_k$ die Zahl der Container des Kunden; die Wahrscheinlichkeit, dass bei $n$ Containern und $K$ gleich wahrscheinlichen Kunden
mindestens ein Kunde genau einen Container hat, ist $1 - \dfrac{n!\,[x^n](e^x - x)^K}{K^n}$.

**Literatur.** Anscombe (1973), *Graphs in Statistical Analysis*, The American Statistician 27(1), 17–21 (Datensätze und Kennzahlen S. 19; Cook-Werte stehen dort nicht); Cook (1977), *Detection of Influential Observation in Linear Regression*, Technometrics 19(1), 15–18 (Gl. 4 und 8, Vergleich mit $F(p, n-p, \alpha)$); James u. a., *An Introduction to Statistical Learning*, 2. Aufl., Abschnitt 3.3.3, S. 99 (Hebel, Gl. 3.37). Die Gleichungen sind hier nachgerechnet (Brüche, Löschen und Neuanpassen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
