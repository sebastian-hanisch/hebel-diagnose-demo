# Hebel und Diagnose – ein Punkt entscheidet – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-hebel-diagnose-demo.streamlit.app/)** -->

Konzepte-Demo zur Regression für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vorgänger unter den Nachbar-Demos ist [ols-projektion-demo](https://github.com/sebastian-hanisch/ols-projektion-demo): Dort ist die Hat-Matrix ein Rechenobjekt (ihre Diagonale heißt Hebel, ihre Spur ist p, der Hebel hängt nur an x), und die Anpassung wird als Ganzes genommen; stillschweigend gilt dabei, dass jeder Punkt für das Ergebnis ungefähr gleich viel zählt. Diese Demo hebt das auf: **Ein einzelner Punkt kann die Gerade bestimmen, und das Residuum zeigt es nicht.** Vehikel ist die Verweilzeit von Export-Containern (Vorlauf als Merkmal), dazu das Anscombe-Quartett und zwei Handbeispiele in Brüchen. Zähleinheit bei allen Anteilen ist ein **Datensatz** (eine Wiederholung); alle Zahlen sind eigene Nachrechnung in diesem Modell, bei diesen Parametern, nicht neu (Anscombe 1973 und Cook 1977 gibt es seit Jahrzehnten); neu ist nur die Vorführbarkeit mit Zähleinheiten.

## Warum dieses Problem

Eine angepasste Gerade fasst alle Punkte zusammen, aber nicht jeder Punkt wiegt gleich. Der **Hebel** h eines Punkts hängt nur an seinem x-Wert und sagt, wie stark sein y die Anpassung zieht; bei h = 1 geht die Gerade genau durch den Punkt. Das **Residuum** zeigt nur den Anteil 1 − h der Abweichung, die man ohne den Punkt gesehen hätte (e = (1 − h)·(y − ŷ₍ᵢ₎)). Die Kernfragen der Demo: Welcher Punkt entscheidet? Warum zeigt das Residuum es nicht? Wann schlägt eine Kennzahl an, und wann zu Unrecht?

## Modell

Einfache Gerade y = b0 + b1·x, p = 2 Spalten (Achsenabschnitt und Steigung), n Punkte. Hebel h = 1/n + (x − x̄)²/Sxx, Spur der Hat-Matrix p, Mittel der Hebel p/n (ISLR2, Abschnitt 3.3.3, S. 99, schreibt (p+1)/n mit p als Zahl der Merkmale ohne Achsenabschnitt: dieselbe Zahl, andere Zählung). s² = SSE/(n − p). **Intern studentisiert** t = e/(s·√(1 − h)), **extern studentisiert** t* = t·√((n − p − 1)/(n − p − t²)) (die Benennung „studentisiertes Residuum“ meint je nach Quelle das eine oder das andere; die Demo schreibt „intern“ und „extern“ dazu). **Cook-Abstand** D = e²·h/(p·s²·(1 − h)²) = t²/p · h/(1 − h) (Cook 1977, Gl. 4 und 8); bei Hebel 1 ist D = 0/0 undefiniert und wird nie als Zahl oder `inf` gezeigt. Schwellen (eine Auswahl, keine Wahrheit): Hebel über 2p/n; Cook über 4/n; Cook über F(p, n − p; 0,5); Cook über 1; t* über 3 im Betrag. **Bei p = 2 sind 2p/n und 4/n dieselbe Zahl.** Die Faustregel 4/n steht nicht bei Cook 1977; Cook vergleicht mit F(p, n − p; α) und sagt für F(·; 0,5), dass Löschen die Schätzung an den Rand des 50-%-Vertrauensellipsoids verschiebt. Formeln und Konventionstabelle stehen im 📐-Expander der App; die Kerne sind `hdi_algorithm.py` (Zahlen, vektorisiert über Datensätze) und `hdi_exact.py` (Brüche).

**Vehikel.** Export-Container: Vorlauf v ~ Gleich(1; 10) Tage, Kunde gleichverteilt aus 12, Verweilzeit d = (0,6 + 0,85·v)·exp(0,3·u_Kunde)·exp(0,25·z) mit z ~ N(0; 1); die zwölf Kundeneffekte u_k sind eine feste Liste. Die Formeln stammen aus dem Export-Zweig des Containerstroms der [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo) und sind hier nachgebaut (kein Import über Repos). Die Gerade d = b0 + b1·v passt im Mittel (E[d | v] = 1,029·(0,6 + 0,85·v), linear in v), die Streuung wächst mit v: das gehört zum Befund (Welt „Verweilzeit“). Zweite Welt „Normal“: gleiches Mittel, konstante Standardabweichung 1,640 (geschlossen gerechnet), normale Fehler. Der Vorlauf des Zusatzpunkts über 10 Tage ist gesetzt („Schiff verschoben“) und kommt im Containerstrom nicht vor. Drei Abweichungen vom Vehikel: das Anscombe-Quartett (vier Datensätze mit exakten Werten), zwei Handbeispiele mit erfundenen Zahlen in Brüchen (damit man von Hand nachrechnen kann; das erste ist in Kundenform erzählt) und Kunden als Faktor in Experiment 5.

## Methodik

Fünf Experimente, alle mit Verdikt aus den Zahlen:

1. **Von Hand und Anscombe.** Sechs Punkte (fünf um y = 4x/5, ein sechster bei x = 6 mit wählbarem y₆) und die vier Anscombe-Sätze, alles in Brüchen bzw. mit exakten Dezimalbrüchen gerechnet. Wählbar ist die Regel, nach der ein Punkt herausgenommen wird (größter Hebel, größtes Cook-D, größtes Residuum).
2. **Ein Zusatzcontainer.** 30 Container plus ein Zusatzpunkt bei gesetztem Vorlauf v* mit Faktor f auf seine Verweilzeit (f = 0,5 halbiert, f = 1 lässt sie). Ein einzelner Datensatz live, daneben die vorgerechnete Tabelle (Median über 4000 Datensätze).
3. **Wann schlägt der Alarm an?** Falschalarm in sauberen Datensätzen gegen Treffer bei halbiertem Zusatzpunkt, je Schwelle und Welt; vorgerechnet (R = 6000 bzw. 4000) und live (R aus dem Regler).
4. **Zwei Hebelpunkte verdecken sich.** Zwei halbierte Zusatzpunkte (Vorlauf 30 und davor) gegen den Punkt bei 30 allein.
5. **Ein Kunde, ein Container.** Im Faktor-Modell „Verweilzeit ~ Kunde“ ist der Hebel 1/n_k; die exakte Wahrscheinlichkeit, dass mindestens ein Kunde genau einen Container hat, gegen Simulation.

**Zufall.** Alle Zufallsdatensätze kommen aus SplitMix64 (`hdi_rng.py`, auf jedem System dieselbe Folge; Normalzahl und Exponentialfunktion ohne die C-Bibliothek). Die ersten R Datensätze eines Zufalls-Seeds sind für jedes R dieselben. **Vorgerechnet** heißt: die Tabellen von Experiment 2 bis 4 stehen in `data/hdi_frozen.json` und werden mit `tools/hdi_make_frozen.py` erzeugt; `--check` rechnet die Probezelle (die ersten 200 Datensätze jeder Tabelle) neu und vergleicht sie mit der Datei (`tests/test_frozen.py`). Die Zahlentabellen dieser README stammen aus derselben Datei (`tools/hdi_readme_tabellen.py`).

## Befunde (gemessen, keine Behauptungen)

Voraussetzungen, soweit nicht anders genannt: einfache Gerade d = b0 + b1·v (OLS, Kunde ignoriert), Export-Container mit Vorlauf Gleich(1; 10), n = 30 Container plus Zusatzpunkt, Zähleinheit Datensatz, Messung dieser Demo mit ihrem Zufallsstrom (Zufalls-Seeds der Tabellen 1101 bis 1103 und 1106 bis 1109). Die Spalte „Test“ nennt, wo die Zahl geprüft wird (Datei `tests/test_claims.py` sofern nicht anders angegeben; „exakt“ heißt in Brüchen gerechnet, „Band“ heißt mit Toleranz über die gemessene Streuung).

**B1. Von Hand: ein Punkt daneben.** Fünf Punkte x = −2 bis 2 mit y = −1, −2, 0, 2, 1 (Gerade y = 4x/5, Residuen 3/5, −6/5, 0, 6/5, −3/5, Hebel 3/5, 3/10, 1/5, 3/10, 3/5), dazu ein sechster Punkt bei x = 6 mit Hebel 19/24 (= 1/6 + 25/40). Bei y₆ = 0 (24/5 unter der Geraden): Steigung von 4/5 auf 1/5, Gerade y = −1/5 + 1/5·x; das Residuum des sechsten Punkts ist −1 und steht **auf Platz 3** der Beträge (2, 8/5, 1, 4/5, 2/5, 1/5), weil e = (1 − 19/24)·(−24/5) = −1: nur 5/24 der Abweichung kommt im Residuum an. Sein Cook-D ist 152/35 = 4,34, das der fünf anderen höchstens 0,30. Test: `test_handbeispiele.py` (exakt, Bruch gleich Bruch, zwei Rechenwege).

**B2. Derselbe Punkt auf der Geraden.** Bei y₆ = 24/5 bleibt die Gerade y = 4x/5, das Residuum ist 0, Cook-D 0; der Standardfehler der Steigung sinkt auf das 0,433-Fache (= (3/20)/(√3/5)) des Wertes der ersten fünf Punkte (s² 9/10, SE² 9/400 gegen 3/25): ein Hebelpunkt, der zur Geraden passt, macht sie genauer. Test: `test_handbeispiele.py` (exakt).

**B3. Anscombe (1973), vier Sätze mit elf Punkten.** In allen vier Sätzen y ≈ 3,00 + 0,50·x, R² 0,666 bis 0,667, Summe der Hebel exakt 2. Cook-Abstand (eigene Rechnung, steht nicht bei Anscombe; p = 2, n = 11, Schwellen 4/n = 0,364, F(2, 9; 0,5) = 0,749, 1): größtes D in Satz I 0,489 (Punkt 3, über 4/n, unter F(.5)), Satz II 0,808 (**die Punkte 6 und 8 mit exakt gleichem Wert**: Gleichstand), Satz III 1,393 (Punkt 3, Hebel 0,236; ohne ihn y = 4,0056 + 0,3454·x), Satz IV 0,137 (Punkt 4); in Satz IV hat der Punkt 8 (x = 19) **Hebel genau 1**, Residuum 0 und kein definiertes D, die übrigen zehn Hebel sind exakt 1/10, und ohne Punkt 8 ist die Steigung nicht bestimmt. Satz II ist eine Kurve (mit quadratischem Term R² = 0,999999), kein Einzelpunkt. Die Faustregel 4/n schlägt schon im Satz I an. Cook-Formel gleich Löschen-und-neu-anpassen, Bruch gleich Bruch, für jeden Punkt mit Hebel unter 1; e = (1 − h)·(y − ŷ₍ᵢ₎) für alle 33 Punkte der Sätze I bis III exakt. Quelle der Zahlen: Anscombe (1973), The American Statistician 27(1), 17–21, Tabelle S. 19 (Scan als Bild gelesen); die Kennzahlen sind nur auf die gedruckten Stellen gleich (Gerade 3,0001 bis 3,0025 im Achsenabschnitt). Test: `test_handbeispiele.py` (exakt), `test_verdicts.py`.

**B4. Ein Zusatzpunkt (Tabelle 1).** Welt „Verweilzeit“, 30 Container plus Zusatzpunkt, **4000 Datensätze**, Median bzw. Anteil. „Alarm“: irgendein Punkt über der Schwelle, die in sauberen Datensätzen 5 % Falschalarm ergibt (größtes Cook-D über 0,844, größtes t* über 5,65; 20 000 saubere Datensätze mit 31 Punkten). Die Anteile „über 4/n“ usw. beziehen sich auf den Zusatzpunkt selbst (Schwellen für n = 31: 4/n = 0,129, F(2, 29; 0,5) = 0,710).

| Vorlauf v* | Faktor f | Hebel | Residuum/s | t* (Betrag) | Cook-D | Cook über 4/n | Cook über F(.5) | t* über 3 | Steigung | SE-Faktor | Alarm Cook | Alarm t* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5,5 | 1,0 | 0,033 | 0,62 | 0,63 | 0,007 | 0,9 % | 0,0 % | 1,1 % | 0,0 % | 0,989 | 5,1 % | 5,0 % |
| 5,5 | 0,5 | 0,033 | 1,65 | 1,73 | 0,048 | 1,7 % | 0,0 % | 2,0 % | 0,0 % | 1,034 | 4,5 % | 4,0 % |
| 10,0 | 1,0 | 0,120 | 1,00 | 1,07 | 0,078 | 37,7 % | 3,4 % | 6,4 % | −1,0 % | 0,958 | 5,1 % | 5,1 % |
| 10,0 | 0,5 | 0,120 | 2,39 | 2,85 | 0,437 | 92,5 % | 17,5 % | 43,2 % | −11,2 % | 1,062 | 11,8 % | 4,0 % |
| 15,0 | 1,0 | 0,332 | 1,13 | 1,41 | 0,477 | 73,7 % | 40,2 % | 14,4 % | −2,2 % | 0,850 | 35,9 % | 6,3 % |
| 15,0 | 0,5 | 0,332 | 2,47 | 3,61 | 2,266 | 98,5 % | 92,7 % | 67,9 % | −26,7 % | 0,986 | 90,6 % | 9,2 % |
| 30,0 | 1,0 | 0,758 | 0,81 | 1,70 | 4,273 | 91,8 % | 79,7 % | 22,9 % | −4,0 % | 0,524 | 77,6 % | 7,8 % |
| 30,0 | 0,5 | 0,758 | 1,63 | 4,24 | 17,488 | 99,4 % | 98,7 % | 78,2 % | −49,0 % | 0,626 | 98,5 % | 22,2 % |

Gelesen: Bei Vorlauf 30 und halbierter Verweilzeit sinkt die Steigung im Median um 49,0 %, das Residuum ist 1,63·s, **fast so groß wie bei einem ebenso halbierten Punkt in der Mitte (Vorlauf 5,5: 1,65·s)**, das Cook-D aber 17,488 gegen 0,048. Das rohe Residuum ist **nicht monoton** im Hebel (1,65 / 2,39 / 2,47 / 1,63 für Vorlauf 5,5 / 10 / 15 / 30), das studentisierte steigt (t* 1,73 / 2,85 / 3,61 / 4,24). Bei Faktor 1 (Zusatzpunkt aus demselben Prozess) bei Vorlauf 30: Steigung −4,0 %, Standardfehler der Steigung ×0,524: ein Hebelpunkt ist nicht deshalb ein Fehler. Der Hebel des Zusatzpunkts überschreitet 2p/n (Tabelle 2) bei Vorlauf 5,5 in keinem Datensatz, bei Vorlauf 10 (Median knapp darunter) in 34,7 %, bei Vorlauf 15 und 30 in allen.

| Vorlauf v* | Hebel (Median) | Hebel über 2p/n |
|---|---|---|
| 5,5 | 0,033 | 0,0 % |
| 10,0 | 0,120 | 34,7 % |
| 15,0 | 0,332 | 100,0 % |
| 30,0 | 0,758 | 100,0 % |

Ein **halbierter Punkt am Rand** (Vorlauf 10, Hebel 0,120) verschiebt die Steigung im Median um 11,2 % nach unten; bei 5 % Falschalarm schlägt aber das Cook-D nur in 11,8 % der Datensätze an, das t* in 4,0 %. In der Welt „Normal“ (gleiches Mittel, konstante Streuung) sind es bei Vorlauf 10: 44,8 % und 15,9 % (Tabelle 3, Steigung −10,8 %); „keine Kennzahl trennt“ gilt also für die Welt „Verweilzeit“ (höchstens 12 %), in der Welt „Normal“ trennt Cook zu 45 %, t* zu 16 %.

| Vorlauf v* | Faktor f | Steigung | SE-Faktor | Alarm Cook | Alarm t* |
|---|---|---|---|---|---|
| 10,0 | 0,5 | −10,8 % | 1,051 | 44,8 % | 15,9 % |
| 30,0 | 0,5 | −46,8 % | 0,620 | 100,0 % | 71,6 % |
| 30,0 | 1,0 | 0,0 % | 0,500 | 60,9 % | 5,0 % |

Test: `test_claims.py` (Tabellen Zeile für Zeile aus der Datei, Probezelle neu gerechnet in `test_frozen.py`).

**B5. Falschalarm in sauberen Datensätzen (Tabelle 4).** n = 30 Container, **6000 Datensätze**, kein Zusatzpunkt; Anteil der Datensätze mit **mindestens einem** Punkt über der Schwelle. „Normal“ heißt gleiches Mittel, konstante Streuung, normale Fehler; „Verweilzeit“ lognormal mit wachsender Streuung.

| Schwelle | Welt Normal | Welt Verweilzeit |
|---|---|---|
| Hebel über 2p/n | 62,1 % | 60,3 % |
| Cook über 4/n | 90,8 % | 97,3 % |
| Cook über F(.5) | 0,4 % | 9,7 % |
| Cook über 1 | 0,0 % | 2,9 % |
| t* (Betrag) über 3 | 17,2 % | 56,0 % |
| Punkte über 4/n je Datensatz (Mittel) | 1,73 | 2,09 |

Die Faustregel 4/n schlägt in 90,8 % (Welt „Normal“) bzw. 97,3 % (Welt „Verweilzeit“) der **sauberen** Datensätze an, im Mittel bei 1,73 bzw. 2,09 Punkten je Datensatz. Orakel: Bei normalen, gleich streuenden Fehlern ist t²/(n − p) Beta(1/2, (n − p − 1)/2)-verteilt (Verteilung der intern studentisierten Residuen, hier nur als Rechenweg benutzt); daraus folgen im Mittel **1,75** Punkte über 4/n (bedingt auf die gemessenen Hebel) gegen 1,73 gemessen (Standardfehler 0,013). Für t* über 3 im Betrag erwartet die t-Verteilung mit 27 Freiheitsgraden 0,172 Punkte je Datensatz gegen 0,176 gemessen. Die 56,0 % bei t* in der Welt „Verweilzeit“ gegen 17,2 % in der Welt „Normal“: ob die wachsende Streuung oder die Schiefe der Lognormalverteilung die Ursache ist, ist **nicht getrennt** (nur beides zusammen gemessen). Test: `test_oracle_hebel_diagnose.py` (Beta- und t-Orakel als Band), `test_claims.py`.

**B6. Zwei Hebelpunkte verdecken sich (Tabelle 5).** Zwei halbierte Zusatzpunkte bei Vorlauf 28 und 30 gegen nur der bei 30; 30 Container, **4000 Datensätze**.

| Größe | Punkt bei 30 allein | Paar (28 und 30) |
|---|---|---|
| Hebel (Median) | 0,756 | 0,390 und 0,462 |
| Cook-D (Median) | 17,45 | 0,68 und 1,04 |
| t* (Betrag, Median) | 4,23 | 1,50 und 1,60 |
| Cook über F(.5), mindestens ein Sonderpunkt | 98,6 % | 92,0 % |
| Cook über F(.5), beide Sonderpunkte | 98,6 % | 15,4 % |
| t* über 3, mindestens ein Sonderpunkt | 79,6 % | 34,4 % |
| t* über 3, beide Sonderpunkte | 79,6 % | 2,7 % |
| Änderung der Steigung (Median) | −48,7 % | −54,1 % |
| Residuum/s des Punkts bei 30 (Median) | 1,65 | 1,15 |

Löschen im Paar, Änderung der Steigung in Standardfehlern der Basis (Median): ohne den Punkt bei 30 0,62, ohne den Punkt bei 28 0,48, ohne beide 4,09.

Im Paar sinkt das Cook-D von 17,45 auf 0,68 und 1,04; beide über F(.5) in 15,4 % statt in 98,6 %, obwohl die Steigung im Paar **stärker** sinkt (−54,1 % gegen −48,7 %). Das Residuum des Punkts bei 30 sinkt von 1,65·s auf 1,15·s. Gilt für das Paar (28, 30) mit Faktor 0,5 bei n = 30, nicht als Satz über Paare; in der App sind Abstand und Faktor frei wählbar (live, R aus dem Regler).

**B7. Ein Kunde, ein Container (Tabelle 6).** Im Faktor-Modell (ein Mittelwert je Kunde, kein Achsenabschnitt) ist der Hebel eines Containers 1/n_k, n_k = Zahl der Container seines Kunden. Exakte Wahrscheinlichkeit (erzeugende Funktion (eˣ − x)¹², ganzzahlig gerechnet; gegen Aufzählung aller Zuordnungen und gegen Simulation geprüft), dass mindestens ein von 12 gleich wahrscheinlichen Kunden genau einen Container hat:

| Container n | mindestens ein Kunde mit genau einem Container |
|---|---|
| 12 | 99,7 % |
| 24 | 98,5 % |
| 36 | 88,0 % |
| 60 | 31,4 % |
| 120 | 0,4 % |

Von Hand (3 Kunden, 4 Container): von 81 Zuordnungen haben 21 keinen Kunden mit genau einem Container, die Wahrscheinlichkeit ist also 20/27. Bei so einem Kunden ist der Hebel exakt 1, das Residuum 0, Cook-D 0/0, und sein Kundeneffekt ist von diesem Container nicht zu trennen. Test: `test_oracle_hebel_diagnose.py` (Aufzählung, Simulation als Band, Hebel über QR), `test_handbeispiele.py`, `test_claims.py`.

**B8. Stabilität der vorgerechneten Zahlen.** Dieselbe Rechnung mit vier weiteren Zufalls-Seed-Versätzen (3000, 7000, 11000, 20000), also fünf Läufen mit der Datei: die Spannweite jedes Anteils der Tabellen ist höchstens 3,3 Prozentpunkte (größte: Alarm des t* in der Welt „Normal“ bei Vorlauf 30, Faktor 0,5), die Spannweite jedes Medians ab 0,1 höchstens 12,4 % des größten Werts (größte: Cook-D des Punkts bei 30 im Paar). Die Tests prüfen deshalb Bänder (Anteile ±4 Prozentpunkte, Mediane ±15 %). Test: `test_claims.py`.

**B9. Orakel.** Die Cook-Formel gleicht dem Löschen und Neuanpassen (Brüche für Anscombe und Handbeispiele; Gleitkomma relativ 1e-9 für Zufallsdatensätze), e = (1 − h)·(y − ŷ₍ᵢ₎) gilt auf 1e-9, die Hat-Matrix über QR hat dieselbe Diagonale (1e-12) und Spur p; t* gleicht e/(s₍ᵢ₎·√(1 − h)) mit s aus der Neuanpassung. Test: `test_oracle_hebel_diagnose.py`.

## Befunde und Korrekturen gegenüber den ersten Rechnungen

- **Neuer Zufallsstrom, neue Zahlen.** Die frühere Messung (Python-Zufall aus numpy, Zufalls-Seeds 1101 bis 1107) und diese Demo (SplitMix64) unterscheiden sich in einzelnen Stellen: Falschalarm der Faustregel 4/n früher 91,2 % („Normal“) und 97,5 % („Verweilzeit“), jetzt 90,8 % und 97,3 %; Cook über F(.5) früher 0,5 % und 9,2 %, jetzt 0,4 % und 9,7 %; t* über 3 früher 17,2 % und 55,7 %, jetzt 17,2 % und 56,0 %; Zusatzpunkt bei Vorlauf 30, Faktor 0,5: Steigung früher −48,6 %, jetzt −49,0 %, Cook-D früher 17,6, jetzt 17,5. **Die README nennt die Zahlen dieser Demo.** (Quelle der früheren Zahlen: die Messdatei `out_rn_v11_hebel.txt` der Vorarbeit, nicht Teil dieses Repos.)
- **„Keine Kennzahl trennt“ galt nach den ersten Rechnungen „höchstens 43 %“ in der Welt „Normal“; gemessen sind 44,8 % (Cook) und 15,9 % (t*).** Die frühere Messung hatte über fünf Zufalls-Seeds 42,9 bis 45,8 % und 15,1 bis 17,2 % gefunden; „höchstens 43 %“ war also zu eng. In der Welt „Verweilzeit“ bleibt es bei höchstens 12 % (11,8 % und 4,0 %).
- **Maskierung:** früher Cook-D im Paar 0,705 und 1,019, beide über F(.5) in 16,4 %, Steigung −53,4 %, Löschen beider 4,04 Standardfehler; jetzt 0,68 und 1,04, 15,4 %, −54,1 %, 4,09. Das Ergebnis (Verdeckung, stärkere Verschiebung im Paar) ist dasselbe.
- **Gepoolte Standardabweichung der Welt „Normal“:** früher 1,638 aus 400 000 Containern gemessen, jetzt 1,640 aus der geschlossenen Formel √(E[m²]·(E[exp(0,6·u)]·e^{2·0,25²} − c²)).
- **Stabilität:** früher „Anteile höchstens 3,6 Prozentpunkte, Mediane ab 0,1 höchstens 5 %“ über fünf Zufalls-Seeds; jetzt Spannweite der Anteile höchstens 3,3 Prozentpunkte, der Mediane ab 0,1 höchstens 12,4 % (die Cook-D-Mediane im Paar streuen mehr, die Zahl 1,04 gilt als „etwa 1“).
- **Das rohe Residuum sinkt nicht einfach mit dem Hebel** (früher für einen einzelnen Datensatz „3,93·s auf 1,72·s“ gemeldet): im Median 1,65·s, 2,39·s, 2,47·s, 1,63·s bei Vorlauf 5,5, 10, 15, 30; das studentisierte steigt. Der exakte Zusammenhang ist e = (1 − h)·(y − ŷ₍ᵢ₎).
- **Verhältnis Cook-D Rand zu Mitte:** das früher für einen einzelnen Datensatz gemessene „bis zum 55-Fachen“ (Zusatzpunkt +6 σ, Zufalls-Seed 1, 20 Punkte) wird hier **nicht** wiederholt; im Median über 4000 Datensätze ist das Verhältnis der Mediane halbiert/Mitte 17,488/0,048, also etwa 360 (anderer Generator, anderer Faktor): beide Zahlen messen verschiedene Dinge und dürfen nicht vermischt werden.
- **Anscombe Satz II:** die Punkte 6 und 8 haben exakt dasselbe Cook-D (Gleichstand); ein Test darf nicht auf `argmax` bauen.

## Ehrliche Grenzen

- Nur die **einfache Gerade** (p = 2) und das Faktor-Modell (Hebel 1/n_k). Hebel bei mehreren Merkmalen (Abstand zum Schwerpunkt im Merkmalsraum), mit Kollinearität und in verallgemeinerten linearen Modellen (gewichtete Hat-Matrix) sind nicht gemessen.
- Die vorgerechneten Tabellen gelten für **n = 30** Container und **einen** Zusatzpunkt (Tabelle 1 bis 3) bzw. ein Paar mit Faktor 0,5 und Abstand 2 (Tabelle 5); Raster: Vorlauf 5,5, 10, 15, 30 und Faktor 1 und 0,5. Zusatzpunkte mit f > 1 (zu lang statt zu kurz), Raster über f und n und größere Gruppen von Hebelpunkten sind nicht vorgerechnet; live lassen sich Faktor, Vorlauf, n und Abstand frei einstellen, die README nennt dafür keine Zahlen.
- Der Vorlauf des Zusatzpunkts über 10 Tage ist **gesetzt**, nicht Teil des Containerstroms.
- Ob die 56,0 % Falschalarm des t* in der Welt „Verweilzeit“ an der wachsenden Streuung oder an der Schiefe liegen, ist nicht getrennt.
- Der Satz „Cook über 4/n schlägt in 97,3 % der sauberen Datensätze an“ gilt für n = 30 und mindestens einen Punkt je Datensatz; je Punkt ist der Anteil klein (im Mittel 2,09 Punkte von 30).
- Der Bruch der Annahmen im Datensatz selbst ist hier gesetzt, nicht gefunden: Welche Container im echten Betrieb Zusatzpunkte mit halber Verweilzeit sind, sagt die Demo nicht.
- Quellenlage: Cook (1977) und ISLR2 sind im Wortlaut gelesen; Anscombe (1973): Zahlen und Kennzahlen der Tabelle (S. 19) als Bild gelesen, die Seiten 20 und 21 nicht angesehen. Die Lizenz der freien Abschriften der Anscombe-Zahlen (Wikipedia, seaborn-data) ist **nicht geprüft**: die Demo führt die Zahlen als Zahlen mit Quellenangabe auf und übernimmt keine Tabelle.

## Verifikation

- **Kern:** Hebel, Cook-D und studentisierte Residuen gegen Handrechnung in Brüchen (`tests/test_handbeispiele.py`, zwei Handbeispiele und das Quartett, mit zwei Rechenwegen), gegen Löschen und Neuanpassen, gegen die QR-Zerlegung, gegen Beta- und t-Verteilung und gegen die Aufzählung der Kundenzuordnungen (`tests/test_oracle_hebel_diagnose.py`); Eigenschaftstests (Verschiebung, Skalierung, Spiegelung, Vertauschung, Zusatzpunkt auf der Geraden; `tests/test_eigenschaften_hebel_diagnose.py`); Randfälle (n = 2, alle x gleich, Hebel genau 1, Gleichstand, Kunde ohne Container; `tests/test_edge_cases.py`).
- **Zufallsstrom:** gleiche Folge in Klasse und Feld, Näherungsformeln für Logarithmus, Kosinus und Exponentialfunktion gleich den exakten Fassungen (`tests/test_rng.py`); Verteilung des Generators gegen feste Referenzwerte des Containerstroms der verweilzeit-demo und gegen geschlossene Momente (Orakel 6).
- **Vorgerechnete Tabellen:** Probezelle neu gerechnet, Form der Datei, Verträglichkeit von Probe und großer Rechnung, Live-Zählung gleich Tabellenzeile bei gleicher Datensatzzahl (`tests/test_frozen.py`).
- **Verdikte:** reine Funktionen, an handgesetzten Werten und an jedem Preset geprüft; ein Preset ohne Effekt meldet keinen (`tests/test_verdicts.py`).
- **Zahlen der README:** jede Zahl dieser README wird in `tests/test_claims.py` gerechnet und im Text gesucht; Tabellen Zeile für Zeile aus `data/hdi_frozen.json`.
- **Presets und Oberfläche:** Presets vollständig und paarweise verschieden, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf; AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum, jede Option der Auswahlfelder) und Footer-Test.
- **Plattform:** exakt nur, was in Brüchen gerechnet ist; alle Gleitkomma-Vergleiche mit Toleranz, Zählungen und Anteile der Neurechnung als Band; Linux-Prüfung in einem Container vor dem ersten Push. Insgesamt 173 Tests.

### Zahl → Test

| Zahl der README | Test | Art |
|---|---|---|
| B1, B2: Brüche der Handbeispiele | `test_handbeispiele.py`, `test_claims.py::test_b1_b2_*` | exakt |
| B3: Anscombe (Hebel, Cook-D, Gleichstand, Satz III ohne Punkt 3) | `test_handbeispiele.py`, `test_claims.py::test_b3_*` | exakt bzw. auf die genannten Stellen |
| B4: Tabellen 1 bis 3 | `test_claims.py::test_b4_*`, `test_frozen.py` | Datei gleich Text, Probe neu gerechnet (Band) |
| B5: Tabelle 4, Beta- und t-Orakel | `test_claims.py::test_b5_*`, `test_oracle_hebel_diagnose.py` | Datei gleich Text, Orakel als Band |
| B6: Tabelle 5 | `test_claims.py::test_b6_*` | Datei gleich Text |
| B7: Tabelle 6, 20/27 | `test_claims.py::test_b7_*`, `test_oracle_hebel_diagnose.py` | exakt (Brüche), Simulation als Band |
| B8: Stabilität | `test_claims.py::test_b8_*` | Neurechnung mit vier anderen Zufalls-Seeds |
| B9: Orakel | `test_oracle_hebel_diagnose.py` | Gleitkomma 1e-9 bis 1e-12 |
| Berichtigungen gegenüber den ersten Rechnungen | `test_claims.py::test_korrekturen_*` | Datei gleich Text |

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `data/hdi_frozen.json` | vorgerechnete Tabellen (Experiment 2 bis 4) |
| `hdi_algorithm.py` | Kern: Gerade, Hebel, Residuen, studentisiert, Cook-D, Schwellen (numpy, über Datensätze vektorisiert) |
| `hdi_constants.py` | Regler-Grenzen, feste Annahmen, Handbeispiele, Anscombe-Zahlen, Farben, Presets |
| `hdi_evaluation.py` | Auswertungen, Verdikte als reine Funktionen, Tabellen für die Anzeige |
| `hdi_exact.py` | Brüche (`fractions`): Gerade, Hebel, Cook-D, Löschen; exakte Wahrscheinlichkeit zu Experiment 5 |
| `hdi_frozen.py` | vorgerechnete Tabellen laden und neu rechnen, Probezelle |
| `hdi_layout.py` | Fester Kopf (Stylesheet) |
| `hdi_measure.py` | Messungen über viele Datensätze (Zusatzpunkt, Falschalarm, Paar) |
| `hdi_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `hdi_rng.py` | SplitMix64 mit Normalzahl und Exponentialfunktion ohne C-Bibliothek |
| `hdi_scenario.py` | Containerstrom (Export-Zweig) und Welt „Normal“, Zufallsströme, Kundenzuordnung |
| `hdi_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `modul_frisch.py` | lädt geänderte Hilfsmodule nach einem Update neu (nur Standardbibliothek) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/hdi_make_frozen.py` | rechnet `data/hdi_frozen.json` und prüft die Probezelle (`--check`) |
| `tools/hdi_readme_tabellen.py` | erzeugt die Zahlentabellen der README aus der Datei |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Verdikte, Zahlen der README, vorgerechnete Tabellen, Presets, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe, Zufallsstrom |

## Bewusst nicht umgesetzt

- **Kein Löschen-Knopf und keine Löschempfehlung.** Einen Punkt nach Ansicht der Daten zu entfernen ist eine Modellwahl, und Tests und Intervalle danach gelten nicht mehr wie vorher (Berk, Brown, Buja, Zhang, Zhao 2013, Annals of Statistics 41, 802–837, laut Quellenbericht der Planung: nach datengetriebener Variablenwahl gelten die klassischen Garantien nicht); gemessen ist das hier nicht.
- **Keine DFFITS, DFBETAS, COVRATIO** und keine robuste Regression. Nach Yu, Yao und Bai (2014, arXiv 1404.6274, laut Quellenbericht der Planung) sind Huber-M-Schätzer unempfindlich gegen ungewöhnliche y-Werte, nicht aber gegen Hebelpunkte in x, MM-Schätzer schützen auch dort; hier nicht gemessen.
- **Keine Mehrfachregression, keine Kollinearität, kein Hebel im verallgemeinerten linearen Modell** (Nachbar-Demos: [kollinearitaet-auswahl-demo](https://github.com/sebastian-hanisch/kollinearitaet-auswahl-demo), [glm-familie-demo](https://github.com/sebastian-hanisch/glm-familie-demo)).
- **Keine robusten oder gewichteten Standardfehler** und kein Streuungsmodell; der Hebel steht dort im Nenner von HC2 und HC3 ([streuung-wls-demo](https://github.com/sebastian-hanisch/streuung-wls-demo)).
- **Kunden als Zufallseffekt und Faktorcodierung** nur in der Fassung „ein Mittelwert je Kunde“ ([cluster-gemischt-demo](https://github.com/sebastian-hanisch/cluster-gemischt-demo), [faktoren-codierung-demo](https://github.com/sebastian-hanisch/faktoren-codierung-demo)).
- **Das Löschen im Zusammenhang mit der Auswahl von Modellen** ([modellwahl-demo](https://github.com/sebastian-hanisch/modellwahl-demo)). Zum Gedanken „ein Punkt entscheidet“ für den Korrelationskoeffizienten siehe [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo); dort geht es um den p-Wert von r, hier um die Verschiebung der Steigung und den Alarm einer Kennzahl.
- **Kein PDF-Export.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechneten Tabellen neu erzeugen: `python tools/hdi_make_frozen.py` (einige Sekunden); Probezelle prüfen: `python tools/hdi_make_frozen.py --check`; README-Tabellen drucken: `python tools/hdi_readme_tabellen.py`.

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
