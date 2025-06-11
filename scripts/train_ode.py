import yaml
import pandas as pd
from utils.data_loader import preprocess_data
from models.ode.model_ode import ode, objective_function, adam_optimizer
from utils.plotting import plot_results

# Load ODE config
with open("config/ode_config.yaml") as f:
    config = yaml.safe_load(f)

file_path = config["file_path"]
years = config["years"]
learning_rate = config["learning_rate"]
n_iter = config["n_iter"]
gauge_params = config["gauge_params"]

# Preprocess data
gauge_ids = list(gauge_params.keys())
df = preprocess_data(file_path, gauge_ids, years)

results = {}

for gauge_id, init_params in gauge_params.items():
    print(f"\nFitting ODE model for Gauge ID: {gauge_id}")
    data = df[df["gauge_id"] == gauge_id]
    rain = data["prcp"].values
    Temp = data["tmax"].values
    obs_Q = data["streamflow"].values
    dt = (data["date"].iloc[1] - data["date"].iloc[0]).days
    T = (data["date"].iloc[-1] - data["date"].iloc[0]).days

    best_params, r2 = adam_optimizer(
        objective_function, rain, Temp, obs_Q, dt, T,
        initial_params=init_params, learning_rate=learning_rate, n_iter=n_iter
    )

    sim_Q = ode(best_params, rain, Temp, dt=dt, T=T)
    plot_results(obs_Q, sim_Q, gauge_id)

    results[gauge_id] = {"Best Parameters": best_params.tolist(), "R2": round(r2, 4)}

print("\nSummary of ODE Results:")
for gid, res in results.items():
    print(f"Gauge {gid} → R²: {res['R2']}, Params: {res['Best Parameters']}")
