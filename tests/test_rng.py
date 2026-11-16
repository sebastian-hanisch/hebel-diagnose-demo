"""Zufallsgenerator: dieselbe Folge in Klasse und Feld, Logarithmus, Kosinus und Exponentialfunktion ohne C-Bibliothek gleich den exakten Fassungen, Momente."""

import numpy as np

import hdi_rng as R


def test_array_stream_equals_class_stream():
    g = R.SplitMix64(12345)
    assert [g.next_u64() for _ in range(50)] == [int(v) for v in R.splitmix_array(12345, 50)]
    g = R.SplitMix64(7)
    assert [g.random() for _ in range(50)] == list(R.uniform_array(7, 50))


def test_known_splitmix_values_for_seed_zero():
    """Referenzfolge von SplitMix64 mit Zustand 0 (Vigna): 0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4."""
    g = R.SplitMix64(0)
    assert g.next_u64() == 0xE220A8397B1DCDAF
    assert g.next_u64() == 0x6E789E6AA1B965F4


def test_below_array_equals_class():
    g = R.SplitMix64(31)
    assert [g.below(12) for _ in range(100)] == list(R.below_array(R.splitmix_array(31, 100), 12))


def test_prefix_property_of_the_counter_stream():
    assert np.array_equal(R.splitmix_array(9, 30), R.splitmix_array(9, 600)[:30])


def test_log_cos_exp_series_match_the_exact_decimal_versions():
    u = np.maximum(R.uniform_array(3, 400), 1e-12)
    ex = np.array([R.exact_log(float(v)) for v in u])
    assert np.max(np.abs(R.log_array(u) - ex) / np.abs(ex).clip(1e-3)) < 1e-14
    ex = np.array([R.exact_cos(2.0 * np.pi * float(v)) for v in u])
    assert np.max(np.abs(R.cos_2pi_array(u) - ex)) < 2e-15
    x = np.linspace(-6.0, 6.0, 241)
    ex = np.array([R.exact_exp(float(v)) for v in x])
    assert np.max(np.abs(R.exp_array(x) / ex - 1.0)) < 1e-15


def test_exp_array_on_special_arguments():
    for x, want in ((0.0, 1.0), (1.0, 2.718281828459045), (-1.0, 0.36787944117144233), (0.25 * 3.0, 2.117000016612675)):
        assert abs(float(R.exp_array(np.array([x]))[0]) / want - 1.0) < 1e-15


def test_normal_array_matches_the_exact_class_value_by_value():
    g = R.SplitMix64(10)
    ex = np.array([g.normal() for _ in range(300)])
    assert np.max(np.abs(R.normal_array(10, 300) - ex)) < 1e-13


def test_normal_from_uniforms_moments():
    z = R.normal_array(5, 200000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1.0) < 0.01          # 200 000 Werte: Standardfehler des Mittels 0,0022
    assert abs(np.mean(z ** 3)) < 0.03 and abs(np.mean(z ** 4) - 3.0) < 0.1
