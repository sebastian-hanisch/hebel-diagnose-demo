"""Rechnet die vorgerechneten Tabellen (Experiment 2 bis 4) und schreibt `data/hdi_frozen.json`.

Aufruf (Python mit numpy und scipy, aus dem Repo-Ordner):
    python tools/hdi_make_frozen.py            # rechnet alles neu (einige Sekunden) und schreibt die Datei
    python tools/hdi_make_frozen.py --check    # rechnet nur die Probezelle neu und vergleicht sie mit der Datei (Exit-Code 1 bei Abweichung)
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hdi_frozen as Z  # noqa: E402


def vergleiche(a, b, pfad="", rel=1e-9, band=0.0):
    """Liste der Abweichungen zwischen zwei verschachtelten Strukturen (Zahlen relativ `rel`, Anteile zusätzlich innerhalb `band`)."""
    fehler = []
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{pfad}: Schlüssel verschieden"]
        for k in a:
            fehler += vergleiche(a[k], b[k], f"{pfad}/{k}", rel, band)
    elif isinstance(a, list):
        if len(a) != len(b):
            return [f"{pfad}: Länge {len(a)} gegen {len(b)}"]
        for i, (x, y) in enumerate(zip(a, b)):
            fehler += vergleiche(x, y, f"{pfad}[{i}]", rel, band)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > max(band, rel * max(abs(a), abs(b), 1e-300)):
            fehler.append(f"{pfad}: {a!r} gegen {b!r}")
    elif a != b:
        fehler.append(f"{pfad}: {a!r} gegen {b!r}")
    return fehler


def main(argv):
    if "--check" in argv:
        frozen = Z.laden()
        neu = Z.probe(frozen)
        # Zahlen aus Gleitkomma-Summen (BLAS/SIMD) dürfen in der letzten Stellen abweichen; Anteile ändern sich nur durch Schwellengleichstände (höchstens 2 Datensätze)
        fehler = vergleiche(neu, frozen["probe"], rel=1e-7, band=2.0 / neu["R"])
        for f in fehler:
            print("ABWEICHUNG", f)
        print("Probezelle passt zur Datei." if not fehler else f"{len(fehler)} Abweichungen.")
        return 1 if fehler else 0
    t = time.time()
    frozen = Z.rechne()
    frozen["probe"] = Z.probe(frozen)
    Z.DATA.parent.mkdir(exist_ok=True)
    Z.DATA.write_text(json.dumps(frozen, indent=1, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    print(f"geschrieben: {Z.DATA} ({time.time() - t:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
