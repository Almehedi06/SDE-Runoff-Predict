import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def ode(par, rain, Temp, dt=1, T=None):
    """
    Deterministic ODE runoff model.
    """
    if T is None:
        T = len(rain)
    n = int(T / dt)
    x = np.zeros(n)
    Qeff = np.zeros(n)
    A, b, k, Qref, bf = par
    epsilon = 1e-6

    for i in range(n - 1):
        try:
            x[i + 1] = max(0, bf + x[i] +
                (rain[i] - Qref * ((x[i] + epsilon) / (k + epsilon)) ** (1 / (2 - b))) * dt)
            x[i + 1] = min(x[i + 1], 1e6)
        except OverflowError:
            x[i + 1] = 0

    try:
        Qeff[:] = A * (Qref * ((x + epsilon) / (k + epsilon)) ** (1 / (2 - b)))
    except OverflowError:
        Qeff[:] = np.nan

    Qeff[np.isnan(Qeff) | np.isinf(Qeff)] = 0
    return Qeff


def objective_function(params, rain, Temp, observed_discharge, dt, T):
    """
    Objective function for ODE optimization.
    """
    modeled_discharge = ode(params, rain, Temp, dt=dt, T=T)
    if np.any(np.isnan(modeled_discharge)) or np.any(np.isinf(modeled_discharge)):
        return np.inf
    return mean_squared_error(observed_discharge[:len(modeled_discharge)], modeled_discharge)


def adam_optimizer(objective_function, rain, Temp, observed_discharge, dt, T,
                   initial_params, learning_rate, n_iter,
                   beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Adam optimizer to minimize ODE objective.
    """
    params = np.array(initial_params)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    t = 0

    for _ in range(n_iter):
        t += 1
        grad = np.zeros_like(params)
        epsilon_grad = 1e-5

        for i in range(len(params)):
            params_eps = np.array(params)
            params_eps[i] += epsilon_grad
            loss_eps = objective_function(params_eps, rain, Temp, observed_discharge, dt, T)
            loss = objective_function(params, rain, Temp, observed_discharge, dt, T)
            grad[i] = (loss_eps - loss) / epsilon_grad

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    modeled_discharge = ode(params, rain, Temp, dt=dt, T=T)
    final_r2 = r2_score(observed_discharge[:len(modeled_discharge)], modeled_discharge)
    return params, final_r2
