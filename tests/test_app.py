"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Ergänzungen zur Vorlage: Auswahlfelder, Zahlenfeld, Presets mit ihren Verdikten -----------------------------------------------------------------

def test_every_choice_widget_option_does_not_raise():
    """Jede Option jedes Auswahlfelds (radio, selectbox) einzeln: keine Ausnahme."""
    at = _fresh_app()
    felder = [(art, w.key, list(w.options)) for art in ("radio", "selectbox") for w in getattr(at, art) if w.key]
    assert felder, "keine Auswahlfelder gefunden"
    for art, key, optionen in felder:
        for option in optionen:
            at = _fresh_app()
            getattr(at, art)(key=key).set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e) for e in at.exception])


def test_number_input_zufalls_seed_at_min_and_max():
    for wert in (0, 999999):
        at = _fresh_app()
        at.number_input(key="seed_input").set_value(wert).run(timeout=TIMEOUT)
        assert not at.exception, (wert, [str(e) for e in at.exception])


def test_standard_zeigt_die_kernfragen_und_keine_fehlermeldung():
    at = _fresh_app()
    texte = [m.value for m in at.markdown]
    assert any("Kernfragen" in t for t in texte)
    assert not at.error
    assert len(at.warning) + len(at.info) + len(at.success) >= 8          # Kernfragen, Experiment 1 bis 5 mit Verdikten


def test_preset_zwei_hebelpunkte_zeigt_die_verdeckung_im_verdikt():
    at = _fresh_app()
    at.sidebar.button(key="preset_Zwei Hebelpunkte").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("verdecken sich gegenseitig" in w.value for w in at.warning)


def test_preset_sauberer_datensatz_meldet_keine_wirkung():
    at = _fresh_app()
    at.sidebar.button(key="preset_Sauberer Datensatz").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("ändert der Zusatzpunkt die Steigung nur um" in i.value for i in at.info)
    assert not any("verschiebt der Zusatzpunkt die Steigung" in w.value for w in at.warning)


def test_anscombe_iv_preset_nennt_hebel_genau_1():
    at = _fresh_app()
    at.sidebar.button(key="preset_Anscombe Satz IV").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Hebel genau 1" in w.value for w in at.warning)


def test_preset_setzt_nicht_genannte_regler_auf_den_standard():
    at = _fresh_app()
    at.number_input(key="seed_input").set_value(555).run(timeout=TIMEOUT)
    at.sidebar.button(key="preset_Kunde mit einem Container").click().run(timeout=TIMEOUT)
    assert at.number_input(key="seed_input").value == 21 and at.slider(key="n_zuordnung").value == 36
