import numpy as np
import pandas as pd
import sys

def check(npz_file, csv_file):
    # Check Config
    data = np.load(npz_file, allow_pickle=True)
    cfg = data['config'].item()
    print(f"Config R_max: {cfg['pde']['R_max']}")
    print(f"Config theta_0: {cfg['spike_gen']['theta_0']}")
    
    # Check CSV
    df = pd.read_csv(csv_file)
    leak = df['leak'].values
    time = df['time'].values
    mass = df['mass'].values
    
    # When does leak start?
    nonzero = np.where(leak > 1e-6)[0]
    if len(nonzero) > 0:
        t_start = time[nonzero[0]]
        print(f"Leak starts at t = {t_start:.4f} s.")
        print(f"Max Leak: {np.max(leak)}")
        print(f"Mass at Leak Start: {mass[nonzero[0]]}")
    else:
        print("No leakage > 1e-6 detected.")
        
    print(f"Mass[0]: {mass[0]}")
    print(f"Mass[-1]: {mass[-1]}")

if __name__ == "__main__":
    check("results/pop_sim_check.npz", "results/pop_sim_check_mass_debug.csv")
