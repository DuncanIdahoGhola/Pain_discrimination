#%% 

import numpy as np
from scipy.stats import t


def paired_tost_power(
    n,
    equivalence_bound,
    true_effect=0.0,
    alpha=0.05,
    simulations=500_000,
    seed=42
):
    """
    Monte Carlo power simulation for paired-samples TOST.

    All effect sizes and equivalence bounds are expressed
    in Cohen's dz units.
    """

    rng = np.random.default_rng(seed)

    df = n - 1
    critical_t = t.ppf(1 - alpha, df)

    # Generate standardized sampling distributions.
    # The sample mean and sample variance are independent
    # under the normality assumption.

    z = rng.standard_normal(simulations)
    chi = rng.chisquare(df, simulations)

    # Sample mean and sample SD, assuming population SD = 1
    sample_mean = true_effect + z / np.sqrt(n)
    sample_sd = np.sqrt(chi / df)

    # Standardized TOST statistics
    t_lower = (
        (sample_mean / sample_sd) + equivalence_bound
    ) * np.sqrt(n)

    t_upper = (
        (sample_mean / sample_sd) - equivalence_bound
    ) * np.sqrt(n)

    # Both one-sided tests must be significant
    equivalent = (
        (t_lower > critical_t) &
        (t_upper < -critical_t)
    )

    return np.mean(equivalent)


def find_required_bound(
    n,
    target_power=0.80,
    true_effect=0.0,
    alpha=0.05,
    simulations=500_000,
    seed=42
):
    """Find the smallest dz equivalence bound achieving target power."""

    low = 0.0
    high = 2.0

    while paired_tost_power(
        n, high, true_effect, alpha, simulations, seed
    ) < target_power:
        high *= 2

    for _ in range(40):
        mid = (low + high) / 2

        power = paired_tost_power(
            n, mid, true_effect, alpha, simulations, seed
        )

        if power >= target_power:
            high = mid
        else:
            low = mid

    return high


def run_analysis(
    sample_sizes=(51, 73),
    previous_bound=0.45,
    target_power=0.80,
    true_effect=0.0,
    alpha=0.05,
    simulations=500_000
):

    print("\n========================================")
    print("     PAIRED-SAMPLES TOST POWER ANALYSIS")
    print("========================================")

    print(f"Target power: {target_power * 100:.0f}%")
    print(f"Alpha: {alpha}")
    print(f"Assumed true effect (dz): {true_effect}")
    print(f"Previous equivalence bound: ±{previous_bound}")
    print(f"Simulations: {simulations:,}")

    results = []

    for n in sample_sizes:

        previous_power = paired_tost_power(
            n=n,
            equivalence_bound=previous_bound,
            true_effect=true_effect,
            alpha=alpha,
            simulations=simulations
        )

        required_bound = find_required_bound(
            n=n,
            target_power=target_power,
            true_effect=true_effect,
            alpha=alpha,
            simulations=simulations
        )

        result = {
            "n": n,
            "previous_bound": previous_bound,
            "previous_power": previous_power,
            "required_bound": required_bound,
            "target_power": target_power
        }

        results.append(result)

        print(f"\n------------ N = {n} ------------")

        print(
            f"Power with ±{previous_bound}: "
            f"{previous_power * 100:.2f}%"
        )

        print(
            f"Required bound for {target_power * 100:.0f}% power: "
            f"±{required_bound:.4f} dz"
        )

        print(
            f"Achieved power: "
            f"{paired_tost_power(n, required_bound, true_effect, alpha, simulations) * 100:.2f}%"
        )

    return results


# ==========================================
# MODIFY YOUR PARAMETERS HERE
# ==========================================

results = run_analysis(
    sample_sizes=(31, 51, 63),
    previous_bound=0.45,
    target_power=0.80,
    true_effect=0.0,
    alpha=0.05,
    simulations=500_000
)
