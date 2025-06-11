# Placeholder for future comparative evaluation of ODE vs SDE vs other ML models
# Could include statistical tests, visual summaries, or ranking logic

def compare_models(results_ode, results_sde):
    """
    Compare performance metrics of ODE and SDE models.
    """
    for gid in results_ode:
        r2_ode = results_ode[gid]["R2"]
        r2_sde = results_sde[gid]["R2"]
        print(f"Gauge {gid} → ΔR²: {r2_sde - r2_ode:.3f} (SDE - ODE)")
