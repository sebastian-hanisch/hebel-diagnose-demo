"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import hdi_constants as C
from hdi_evaluation import Settings


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None
    field: str = ""             # Feld von Settings


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, field="seed"),
    "n_container": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX, field="n"),
    "r_live": SettingSpec("r", int, C.DEFAULT_R, C.R_MIN, C.R_MAX, field="r"),
    "y6_fuenftel": SettingSpec("y6", int, C.DEFAULT_Y6, C.Y6_MIN, C.Y6_MAX, field="y6"),
    "punktregel": SettingSpec("regel", str, C.DEFAULT_PUNKTREGEL, options=C.PUNKTREGELN, field="regel"),
    "anscombe_satz": SettingSpec("satz", str, C.DEFAULT_SATZ, options=C.SAETZE, field="satz"),
    "v_stern": SettingSpec("v", float, C.DEFAULT_V, C.V_MIN, C.V_MAX, field="v"),
    "f_faktor": SettingSpec("f", float, C.DEFAULT_F, C.F_MIN, C.F_MAX, field="f"),
    "schwelle": SettingSpec("schwelle", str, C.DEFAULT_SCHWELLE, options=C.SCHWELLEN, field="schwelle"),
    "welt": SettingSpec("welt", str, C.DEFAULT_WELT, options=C.WELTEN, field="welt"),
    "v_alarm": SettingSpec("va", float, C.DEFAULT_V_ALARM, options=C.V_ALARM_WERTE, field="va"),
    "abstand": SettingSpec("abstand", float, C.DEFAULT_ABSTAND, C.ABSTAND_MIN, C.ABSTAND_MAX, field="abstand"),
    "f_paar": SettingSpec("fp", float, C.DEFAULT_F, C.F_MIN, C.F_MAX, field="fp"),
    "n_zuordnung": SettingSpec("nz", int, C.DEFAULT_NZ, C.NZ_MIN, C.NZ_MAX, field="nz"),
}
PRESET_KEYS = {spec.url_param: state_key for state_key, spec in SETTING_SPECS.items()}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def begrenze(spec: SettingSpec, value):
    """Wert auf die Grenzen bzw. Optionen des Reglers bringen; None, wenn er nicht passt."""
    if spec.options is not None:
        return value if value in spec.options else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = begrenze(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler setzen: die im Preset genannten auf dessen Wert, alle übrigen auf ihren Standard (so ist das Preset danach genau der aktuelle Zustand)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(spec.url_param, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (setzt alle Regler, nicht genannte auf den Standard):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def current_settings() -> Settings:
    """Settings aus dem Session State (alle Regler sind initialisiert)."""
    return Settings(**{spec.field: spec.caster(st.session_state.get(state_key, spec.default)) for state_key, spec in SETTING_SPECS.items()})
