import numpy as np

def sde(par, rain, Temp, runs=1000, dt=1, T=None):
    """
    Stochastic Differential Equation (SDE) model for runoff.
    """
    if T is None:
        T = len(rain)

    n = T
    x = np.zeros((n, runs))
    Qeff = np.ones((n, runs))
    A, b, k, Qref, bf = par
    epsilon = 1e-6

    for j in range(runs):
        for i in range(n - 1):
            noise_term = np.random.normal(0, 7 * np.sqrt(dt))
            x[i + 1, j] = max(
                0,
                bf + x[i, j] +
                (rain[i] - Qref * ((x[i, j] + epsilon) / (k + epsilon)) ** (1 / (2 - b))) * dt +
                noise_term
            )

    Qeff[:] = A * (Qref * ((x + epsilon) / (k + epsilon)) ** (1 / (2 - b)))
    return Qeff


def compute_coverage_probability(observed_discharge, modeled_discharge_all, confidence_level=0.95):
    """
    Calculate coverage probability of observed discharge within ensemble.
    """
    lower_bound = np.percentile(modeled_discharge_all, (1 - confidence_level) / 2 * 100, axis=1)
    upper_bound = np.percentile(modeled_discharge_all, (1 + confidence_level) / 2 * 100, axis=1)

    within_bounds = (observed_discharge >= lower_bound) & (observed_discharge <= upper_bound)
    return np.mean(within_bounds)
