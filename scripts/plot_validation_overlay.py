import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.ndimage import gaussian_filter1d

def plot_overlay(npz_file, out_png):
    data = np.load(npz_file)
    A_mc = data['A']
    # PDE?
    if 'A_pde' not in data:
        print("No A_pde in file.")
        return
    A_pde = data['A_pde']
    
    dt = 0.0001
    time = np.arange(len(A_mc)) * dt
    
    # Smooth (50ms)
    sigma = 0.05 / dt
    A_mc_sm = gaussian_filter1d(A_mc, sigma)
    A_pde_sm = gaussian_filter1d(A_pde, sigma)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time, A_mc, 'k-', alpha=0.1, label='MC Raw')
    plt.plot(time, A_mc_sm, 'k-', linewidth=2, label='MC Smoothed')
    plt.plot(time[::10], A_pde_sm[::10], 'r--', linewidth=2, label='PDE Smoothed')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Activity (Hz)')
    plt.title('Validation Overlay (Uncoupled/Low Coupling)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(out_png)
    print(f"Saved {out_png}")

if __name__ == "__main__":
    plot_overlay(sys.argv[1], sys.argv[2])
