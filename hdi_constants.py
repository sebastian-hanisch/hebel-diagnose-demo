"""Konstanten: Regler-Grenzen, feste Annahmen, Datensätze von Hand, Anscombe-Zahlen, Farben und Presets.

Vehikel: der Export-Zweig des Containerstroms der `verweilzeit-demo` (Formeln hier nachgebaut, mit Herkunftsvermerk in `hdi_scenario.py`):
Vorlauf v ~ Gleich(1; 10) Tage, Kunde gleichverteilt aus 12, Verweilzeit d = (0,6 + 0,85·v)·exp(0,3·u_Kunde)·exp(0,25·z) mit z ~ N(0; 1)."""

# --- Regler ----------------------------------------------------------------------------------------------------------------------------------

SEED_MAX, DEFAULT_SEED = 999999, 21
N_MIN, N_MAX, DEFAULT_N = 10, 60, 30            # Container im Datensatz (ohne Zusatzpunkt)
R_MIN, R_MAX, DEFAULT_R = 20, 1000, 200         # Datensätze der Live-Zählung (Experimente 3 und 4)
Y6_MIN, Y6_MAX, DEFAULT_Y6 = 0, 24, 0           # y6 = Wert/5; 24/5 liegt auf der Geraden der ersten fünf Punkte
V_MIN, V_MAX, DEFAULT_V = 1.0, 40.0, 15.0       # Vorlauf des Zusatzpunkts in Tagen (über 10 ist gesetzt, nicht Teil des Containerstroms)
F_MIN, F_MAX, DEFAULT_F = 0.25, 2.0, 0.5        # Faktor auf die Verweilzeit des Zusatzpunkts (0,5 = halbiert, 1 = unverändert)
ABSTAND_MIN, ABSTAND_MAX, DEFAULT_ABSTAND = 0.5, 10.0, 6.0
NZ_MIN, NZ_MAX, DEFAULT_NZ = 12, 120, 24        # Container der Kundenzuordnung (Experiment 5)

PUNKTREGELN = ("Größter Hebel", "Größtes Cook-D", "Größtes Residuum")
DEFAULT_PUNKTREGEL = "Größtes Cook-D"
SAETZE = ("I", "II", "III", "IV")
DEFAULT_SATZ = "III"
WELTEN = ("Verweilzeit", "Normal")
DEFAULT_WELT = "Verweilzeit"
SCHWELLEN = (
    "Hebel über 2p/n",
    "Cook-D über 4/n",
    "Cook-D über F(p, n−p; 0,5)",
    "Cook-D über 1",
    "|t*| über 3",
    "5-%-Schwelle (Datensatzalarm)",
)
DEFAULT_SCHWELLE = SCHWELLEN[1]
V_ALARM_WERTE = (5.5, 10.0, 15.0, 30.0)         # Raster der vorgerechneten Tabelle
DEFAULT_V_ALARM = 30.0
F_ALARM_WERTE = (1.0, 0.5)

P = 2                                            # Spalten der Entwurfsmatrix: Achsenabschnitt und Steigung

# --- Containerstrom (Export-Zweig) -----------------------------------------------------------------------------------------------------------

K_KUNDEN = 12
U_SCALE_EXPORT = 0.3                             # Kundeneffekt im Export: ein Drittel der Importstärke
S_EXPORT = 0.25                                  # Streuung der log-Verweilzeit im Export
M0, M1 = 0.6, 0.85                               # Median der Verweilzeit: 0,6 + 0,85·v Tage
V_LO, V_HI = 1.0, 10.0
# Die zwölf Kundeneffekte u_k (log-Skala, Streuung 0,30) sind eine feste Liste, aus dem Containerstrom der verweilzeit-demo übernommen und nicht neu gezogen.
UK = (
    -0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
    -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771,
)
EXTRA_PUNKTE = 2                                 # je Datensatz zwei Reservepunkte im Zufallsstrom (Zusatzpunkt, zweiter Zusatzpunkt)

# --- Handbeispiele und Anscombe-Quartett (Zahlen, keine Tabellenübernahme) --------------------------------------------------------------------

# Von Hand B: fünf Basispunkte plus ein sechster bei x = 6 (y6 wählbar). Basisgerade y = 4x/5.
HAND_B_X = (-2, -1, 0, 1, 2)
HAND_B_Y = (-1, -2, 0, 2, 1)
HAND_B_X6 = 6
# Von Hand A: drei Container von Kunde A (x = 0), einer von Kunde B (x = 4).
HAND_A_X = (0, 0, 0, 4)
HAND_A_Y = (1, 2, 3, 6)

# Anscombe (1973): vier Datensätze zu je elf Punkten; Satz I bis III teilen die x-Werte. Die y-Werte stehen in Hundertsteln (804 heißt 8,04), damit sie exakte Brüche sind.
ANSCOMBE_X123 = "10 8 13 9 11 14 6 4 12 7 5"
ANSCOMBE_NENNER = 100
ANSCOMBE = {
    "I": (ANSCOMBE_X123, "804 695 758 881 833 996 724 426 1084 482 568"),
    "II": (ANSCOMBE_X123, "914 814 874 877 926 810 613 310 913 726 474"),
    "III": (ANSCOMBE_X123, "746 677 1274 711 781 884 608 539 815 642 573"),
    "IV": ("8 8 8 8 8 8 8 19 8 8 8", "658 576 771 884 847 704 525 1250 556 791 689"),
}
ANSCOMBE_QUELLE = "Anscombe (1973), The American Statistician 27(1), 17–21; die Zahlen sind hier als Zahlen aufgeführt, die App rechnet alles aus ihnen"

# --- Vorgerechnete Tabellen ------------------------------------------------------------------------------------------------------------------

FROZEN_FILE = "hdi_frozen.json"
R_FROZEN_B, R_FROZEN_C, R_FROZEN_D, R_FROZEN_THR = 4000, 6000, 4000, 20000
R_PROBE = 200                                    # Datensätze der Probezelle: der Test rechnet sie neu und vergleicht mit der Datei
SEED_B_VZ, SEED_B_NORMAL, SEED_THR_VZ, SEED_THR_NORMAL = 1101, 1106, 1107, 1108
SEED_C_VZ, SEED_C_NORMAL, SEED_D = 1102, 1109, 1103
N_FROZEN = 30                                    # die Tabellen gelten für 30 Container (plus Zusatzpunkt)

# --- Aussehen --------------------------------------------------------------------------------------------------------------------------------

COLOR_POINT = "rgba(120,120,120,0.85)"
COLOR_SPECIAL = "#d1495b"
COLOR_FIT = "#2F6B65"
COLOR_LOO = "#e08e2b"
COLOR_FIT2 = "#4c78a8"
COLOR_BAR = "#4c78a8"

# --- Presets ---------------------------------------------------------------------------------------------------------------------------------

PRESETS = {
    "Von Hand: sechster Punkt daneben": {"y6": DEFAULT_Y6},
    "Von Hand: sechster Punkt auf der Geraden": {"y6": 24},
    "Anscombe Satz IV": {"satz": "IV", "regel": "Größter Hebel"},
    "Halbierter Hebelpunkt (Vorlauf 30)": {"v": 30.0, "f": 0.5},
    "Hebelpunkt aus demselben Prozess": {"v": 30.0, "f": 1.0},
    "Halbierter Punkt am Rand (Vorlauf 10)": {"v": 10.0, "f": 0.5, "va": 10.0, "schwelle": SCHWELLEN[5]},
    "Zwei Hebelpunkte": {"abstand": 2.0, "fp": 0.5},
    "Sauberer Datensatz": {"v": 5.5, "f": 1.0},
    "Kunde mit einem Container": {"nz": 36},
}
PRESET_HELP = {
    "Von Hand: sechster Punkt daneben": "Fünf Punkte auf einer Geraden mit Rauschen, ein sechster bei x = 6 mit y = 0, 24/5 unter der Geraden. Wirkt im Ergebnis (Experiment 1, oben).",
    "Von Hand: sechster Punkt auf der Geraden": "Derselbe sechste Punkt, jetzt genau auf der Geraden der ersten fünf: Gerade unverändert, Residuum 0, Cook-D 0. Wirkt in Experiment 1.",
    "Anscombe Satz IV": "Alle Punkte außer einem haben x = 8; der elfte steht bei x = 19 und hat Hebel genau 1. Wirkt in Experiment 1 (Anscombe).",
    "Halbierter Hebelpunkt (Vorlauf 30)": "Ein Zusatzcontainer mit Vorlauf 30 Tagen (gesetzt, Schiff verschoben) und halber Verweilzeit. Wirkt in Experiment 2.",
    "Hebelpunkt aus demselben Prozess": "Derselbe Zusatzcontainer, aber mit der Verweilzeit, die der Prozess liefert (Faktor 1): ein Hebelpunkt ist kein Fehler. Wirkt in Experiment 2.",
    "Halbierter Punkt am Rand (Vorlauf 10)": "Halbierter Zusatzcontainer am Rand des beobachteten Vorlaufs: verschiebt die Steigung, schlägt aber bei 5 % Falschalarm kaum an. Wirkt in Experiment 2 und 3.",
    "Zwei Hebelpunkte": "Zwei halbierte Zusatzcontainer bei Vorlauf 28 und 30: jeder verdeckt den anderen. Wirkt in Experiment 4.",
    "Sauberer Datensatz": "Zusatzcontainer in der Mitte des Vorlaufs mit der Verweilzeit des Prozesses: nichts Besonderes, die Kennzahlen bleiben unauffällig. Wirkt in Experiment 2.",
    "Kunde mit einem Container": "36 Container auf 12 Kunden: wie oft hat mindestens ein Kunde genau einen Container (Hebel 1)? Wirkt in Experiment 5.",
}
