import numpy as np
import pytest

from src.simulation import (
    BASELINE,
    sample_claim_amounts,
    sample_claim_counts,
    simulate_annual_losses,
    summarize_annual_losses,
)


def test_zero_expected_claims_gives_zero_in_every_year():
    counts = sample_claim_counts(0, 10, np.random.default_rng(42))

    np.testing.assert_array_equal(counts, np.zeros(10, dtype=int))


def test_sample_mean_is_close_to_expected_claims():
    expected_claims = 50
    n_years = 10_000

    counts = sample_claim_counts(
        expected_claims, n_years, np.random.default_rng(42)
    )

    standard_error = np.sqrt(expected_claims / n_years)

    assert counts.shape == (n_years,)
    assert counts.mean() == pytest.approx(
        expected_claims, abs=5 * standard_error
    )

def test_zero_claims_gives_an_empty_array():
    amounts = sample_claim_amounts(5_000, 7_500, 0, np.random.default_rng(42))

    assert amounts.shape == (0,)


def test_claim_amounts_match_severity_assumptions():
    mean = 5_000
    std_dev = 7_500
    n_claims = 100_000

    amounts = sample_claim_amounts(
        mean, std_dev, n_claims, np.random.default_rng(42)
    )

    assert amounts.shape == (n_claims,)
    assert np.all(amounts > 0)

    mean_standard_error = std_dev / np.sqrt(n_claims)
    assert amounts.mean() == pytest.approx(mean, abs=5 * mean_standard_error)
    assert amounts.std(ddof=1) == pytest.approx(std_dev, rel=0.10)

def test_annual_losses_are_nonnegative_and_reproducible():
    inputs = {**BASELINE, "n_years": 100}

    losses = simulate_annual_losses(**inputs)
    repeated_losses = simulate_annual_losses(**inputs)

    assert losses.shape == (100,)
    assert np.all(losses >= 0)
    np.testing.assert_array_equal(losses, repeated_losses)


def test_zero_frequency_gives_zero_annual_losses():
    inputs = {**BASELINE, "frequency": 0, "n_years": 10}

    losses = simulate_annual_losses(**inputs)

    np.testing.assert_array_equal(losses, np.zeros(10))


def test_baseline_mean_matches_analytical_expectation():
    losses = simulate_annual_losses(**BASELINE)

    expected_claims = BASELINE["exposure"] * BASELINE["frequency"]
    severity_mean = BASELINE["severity_mean"]
    severity_variance = BASELINE["severity_std_dev"] ** 2

    expected_loss = expected_claims * severity_mean
    annual_variance = expected_claims * (severity_variance + severity_mean**2)
    standard_error = np.sqrt(annual_variance / BASELINE["n_years"])

    assert losses.mean() == pytest.approx(expected_loss, abs=5 * standard_error)

def test_risk_metrics_match_hand_calculated_values():
    losses = np.arange(1, 31) * 100

    summary = summarize_annual_losses(losses, threshold=2_900)

    assert summary["mean"] == pytest.approx(1_550)
    assert summary["median"] == pytest.approx(1_550)
    assert summary["var_95"] == 2_900
    assert summary["tvar_95"] == pytest.approx((3_000 + 0.5 * 2_900) / 1.5)
    assert summary["var_99"] == 3_000
    assert summary["tvar_99"] == pytest.approx(3_000)
    assert summary["exceedance_probability"] == pytest.approx(1 / 30)

def test_negative_frequency_is_rejected():
    inputs = {**BASELINE, "frequency": -0.01, "n_years": 10}

    with pytest.raises(ValueError, match="frequency must be"):
        simulate_annual_losses(**inputs)


@pytest.mark.parametrize(
    ("mean", "std_dev", "error_message"),
    [
        (0, 7_500, "mean must be"),
        (5_000, 0, "std_dev must be"),
    ],
)
def test_nonpositive_severity_parameters_are_rejected(
    mean, std_dev, error_message
):
    with pytest.raises(ValueError, match=error_message):
        sample_claim_amounts(mean, std_dev, 10, np.random.default_rng(42))


def test_empty_annual_losses_are_rejected():
    with pytest.raises(ValueError, match="nonempty one-dimensional array"):
        summarize_annual_losses(np.array([]))