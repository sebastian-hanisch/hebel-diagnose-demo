"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """Vier Punkte (Handbeispiel 1): x = 0, 0, 0, 4 und y = 1, 2, 3, 6. Handrechnung: Gerade y = 2 + x, Hebel 1/3, 1/3, 1/3, 1, Residuen −1, 0, 1, 0."""
    return [F(0), F(0), F(0), F(4)], [F(1), F(2), F(3), F(6)]
