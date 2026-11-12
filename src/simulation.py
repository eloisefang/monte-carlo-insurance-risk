import numpy as np

BASELINE = {
    "exposure": 1_000,
    "frequency": 0.05,
    "severity_mean": 5_000,
    "severity_std_dev": 7_500,
    "n_years": 50_000,
    "seed": 42,
}

def sample_claim_counts(
    expected_claims: float,
    n_years: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one simulated poisson claim count per year."""
    if not np.isfinite(expected_claims) or expected_claims < 0:
        raise ValueError("expected_claims must be finite and nonnegative.")

    if (
        isinstance(n_years, (bool, np.bool_))
        or not isinstance(n_years, (int, np.integer))
        or n_years < 1
    ):
        raise ValueError("n_years must be a positive integer.")

    return rng.poisson(lam=expected_claims, size=n_years)

def sample_claim_amounts(
    mean: float,
    std_dev: float,
    n_claims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one simulated lognormal payment per claim, in CAD."""
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("mean must be finite and positive.")

    if not np.isfinite(std_dev) or std_dev <= 0:
        raise ValueError("std_dev must be finite and positive.")

    if (
        isinstance(n_claims, (bool, np.bool_))
        or not isinstance(n_claims, (int, np.integer))
        or n_claims < 0
    ):
        raise ValueError("n_claims must be a nonnegative integer.")

    log_variance = np.log1p((std_dev / mean) ** 2)
    log_mean = np.log(mean) - log_variance / 2
    log_std_dev = np.sqrt(log_variance)

    return rng.lognormal(mean=log_mean, sigma=log_std_dev, size=n_claims)

def simulate_annual_losses(
    *,
    exposure: float,
    frequency: float,
    severity_mean: float,
    severity_std_dev: float,
    n_years: int,
    seed: int,
) -> np.ndarray:
    """Return total claim payments(combine 2 samplers) in CAD for each simulated year."""
    if not np.isfinite(exposure) or exposure < 0:
        raise ValueError("exposure must be finite and nonnegative.")
    if not np.isfinite(frequency) or frequency < 0:
        raise ValueError("frequency must be finite and nonnegative.")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer.")

    rng = np.random.default_rng(seed)
    claim_counts = sample_claim_counts(exposure * frequency, n_years, rng)
    claim_amounts = sample_claim_amounts(
        severity_mean, severity_std_dev, int(claim_counts.sum()), rng
    )

    annual_losses = np.zeros(n_years)
    start = 0
    for year, count in enumerate(claim_counts):
        end = start + count
        annual_losses[year] = claim_amounts[start:end].sum()
        start = end

    return annual_losses

def summarize_annual_losses(
    losses: np.ndarray,
    threshold: float = 350_000,
) -> dict[str, float]:
    """Summarize annual losses in CAD and report exceedance as a probability."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(losses)) or np.any(losses < 0):
        raise ValueError("losses must be finite and nonnegative.")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative.")

    ordered_losses = np.sort(losses)
    n_years = ordered_losses.size
    summary = {
        "mean": float(losses.mean()),
        "median": float(np.median(losses)),
        "exceedance_probability": float(np.mean(losses > threshold)),
    }

    for confidence in (95, 99):
        # Round up to the one-based VaR rank.
        var_rank = (confidence * n_years + 99) // 100
        summary[f"var_{confidence}"] = float(ordered_losses[var_rank - 1])

        # Count the worst observations in hundredths of a simulated year.
        tail_units = (100 - confidence) * n_years
        full_years, remaining_units = divmod(tail_units, 100)
        boundary_index = n_years - full_years - 1
        tail_total = ordered_losses[boundary_index + 1 :].sum()
        tail_total += (remaining_units / 100) * ordered_losses[boundary_index]
        summary[f"tvar_{confidence}"] = float(tail_total / (tail_units / 100))

    return summary