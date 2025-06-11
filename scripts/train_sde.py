import yaml
import pandas as pd
import numpy as np
from utils.data_loader import preprocess_data
from models.sde.model_sde import sde, compute_coverage_probability
from sklearn.metrics import mean_squared_error, r2_score

# Load SDE config
with open("config/sde_config.yaml") as f:
    config = yaml.safe_load(f)

file_path = config["file_path"]
years = config["years"]
runs = config["runs"]
confidence_interval = config["confidence_interval"]
gauge_params = config["gauge_params"]

gauge_ids = list(gauge_params.keys())
df = preprocess_data(file_path, gauge_ids, years)

results = {}

for gauge_id in gauge_ids:
    print(f"\nSimulating SDE model for Gauge ID: {gauge_id}")
    data = df[df["gauge_id"] == gauge_id].copy()
    rain = data["prcp"].values
    Temp = data["tmax"].values
    obs_Q = data["streamflow"].values

    if len(rain) == 0 or len(obs_Q) == 0:
        print(f"Skipping {gauge_id} due to insufficient data.")
        continue

    dt = (data["date"].iloc[1] - data["date"].iloc[0]).days
    T = len(rain)

    sim_ensemble = sde(gauge_params[gauge_id], rain, Temp, runs=runs, dt=dt, T=T)
    sim_mean = sim_ensemble.mean(axis=1)

    r2 = r2_score(obs_Q[:len(sim_mean)], sim_mean)
    mse = mean_squared_error(obs_Q[:len(sim_mean)], sim_mean)
    cp = compute_coverage_probability(obs_Q[:len(sim_mean)], sim_ensemble, confidence_interval)

    results[gauge_id] = {
        "R2": round(r2, 4),
        "MSE": round(mse, 4),
        "CoverageProb": round(cp, 4)
    }

    print(f"Gauge {gauge_id} → R²: {r2:.3f}, MSE: {mse:.2f}, CP: {cp:.2%}")

print("\nSummary of SDE Results:")
for gid, res in results.items():
    print(f"Gauge {gid} → R²: {res['R2']}, MSE: {res['MSE']}, CP: {res['CoverageProb']}")
