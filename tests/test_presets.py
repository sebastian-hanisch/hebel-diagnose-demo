"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Presets paarweise verschieden, Permalink begrenzt."""

import hdi_constants as C
import hdi_evaluation as ev
import hdi_presets as P


def _vollstaendig(preset):
    """Preset als Zustand aller Regler (fehlende Schlüssel = Standardwert)."""
    return {sk: preset.get(spec.url_param, spec.default) for sk, spec in P.SETTING_SPECS.items()}


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[name] for name in C.PRESETS)


def test_preset_keys_are_known_url_params():
    bekannt = {spec.url_param for spec in P.SETTING_SPECS.values()}
    for name, preset in C.PRESETS.items():
        assert set(preset) <= bekannt, name


def test_preset_values_are_inside_the_slider_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for sk, spec in P.SETTING_SPECS.items():
            wert = preset.get(spec.url_param, spec.default)
            assert P.begrenze(spec, wert) == wert, (name, sk, wert)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**{spec.field: preset_wert for spec, preset_wert in
                          ((spec, C.PRESETS["Von Hand: sechster Punkt daneben"].get(spec.url_param, spec.default)) for spec in P.SETTING_SPECS.values())}) == ev.Settings()


def test_presets_are_pairwise_different_states():
    """Sonst könnte `aktives_preset` zwei Presets nicht unterscheiden."""
    zustaende = [tuple(sorted(_vollstaendig(p).items())) for p in C.PRESETS.values()]
    assert len(set(zustaende)) == len(zustaende)


def test_only_the_first_preset_equals_the_default_state():
    standard = {sk: spec.default for sk, spec in P.SETTING_SPECS.items()}
    gleich = [name for name, p in C.PRESETS.items() if _vollstaendig(p) == standard]
    assert gleich == [next(iter(C.PRESETS))]


def test_setting_specs_cover_every_preset_key_and_settings_field():
    assert {spec.field for spec in P.SETTING_SPECS.values()} == set(ev.Settings.__dataclass_fields__)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_begrenze_klemmt_und_lehnt_unbekannte_optionen_ab():
    assert P.begrenze(P.SETTING_SPECS["n_container"], 999) == C.N_MAX and P.begrenze(P.SETTING_SPECS["n_container"], -5) == C.N_MIN
    assert P.begrenze(P.SETTING_SPECS["welt"], "Unsinn") is None and P.begrenze(P.SETTING_SPECS["welt"], "Normal") == "Normal"
    assert P.begrenze(P.SETTING_SPECS["v_alarm"], 12.0) is None and P.begrenze(P.SETTING_SPECS["v_alarm"], 10.0) == 10.0


def test_each_preset_changes_the_experiment_named_in_its_help():
    """Die Hilfe nennt das Experiment, in dem das Preset wirkt; das Preset ändert mindestens einen Regler genau dieses Experiments."""
    regler_je_experiment = {"Experiment 1": {"y6", "satz", "regel"}, "Experiment 2": {"v", "f"}, "Experiment 3": {"va", "schwelle", "welt"}, "Experiment 4": {"abstand", "fp"}, "Experiment 5": {"nz"}}
    for name, preset in C.PRESETS.items():
        genannt = [e for e in regler_je_experiment if e in C.PRESET_HELP[name]]
        assert genannt, name
        assert any(set(preset) & regler_je_experiment[e] for e in genannt), name

