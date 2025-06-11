import matplotlib.pyplot as plt

def plot_results(observed_discharge, modeled_discharge, gauge_id, save_plot=False):
    plt.figure(figsize=(10, 6))
    plt.plot(observed_discharge[:len(modeled_discharge)], '.', color='black', label='Observed Discharge')
    plt.plot(modeled_discharge, color='red', label='Modeled Discharge')
    plt.title(f'Observed vs Modeled Discharge for Gauge ID: {gauge_id}')
    plt.xlabel('Time Steps')
    plt.ylabel('Discharge (Q/Q_mean)')
    plt.legend()
    if save_plot:
        plt.savefig(f'gauge_{gauge_id}_performance.png', dpi=150)
    plt.show()
