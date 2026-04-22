import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze():
    df = pd.read_csv('results/pop_sim_check_mass_debug.csv')
    
    mass = df['mass'].values
    leak = df['leak'].values
    time = df['time'].values
    
    # Stats
    min_mass = np.min(mass)
    max_mass = np.max(mass)
    max_leak_q = np.max(leak)
    
    print(f"Mass Range: {min_mass:.6f} - {max_mass:.6f}")
    print(f"Max Deviation from 1.0: {np.max(np.abs(mass - 1.0)):.6e}")
    print(f"Max Leakage q(R_max): {max_leak_q:.6e}")
    
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax[0].plot(time, mass, label='Mass(t)')
    ax[0].axhline(1.0, color='r', linestyle='--')
    ax[0].set_ylabel('Mass')
    ax[0].set_title(f'PDE Mass Conservation (Rmax=100s)')
    ax[0].grid(True)
    
    # Plot LOG deviation if small
    # ax[0].plot(time, np.abs(mass-1.0), label='|Mass-1|')
    
    ax[1].plot(time, leak, color='orange', label='q(Rmax)')
    ax[1].set_ylabel('Leakage density')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_yscale('log')
    ax[1].grid(True)
    
    plt.savefig('results/figures/Fig_A6_Mass.png')
    print("Saved results/figures/Fig_A6_Mass.png")

if __name__ == "__main__":
    analyze()
