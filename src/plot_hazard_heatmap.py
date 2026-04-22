import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from metrics import ActivityMetrics

def plot_heatmap(npz_file, out_png):
    data = np.load(npz_file)
    if 'rho_field' not in data:
        print("No rho_field in data.")
        return
        
    rho_field = data['rho_field'] # [Time, R]
    r_grid = data['r_grid']
    
    # Downsampled time?
    # Sim stored every 100th step. dt=0.0001 -> 0.01s (10ms) per row.
    
    plt.figure(figsize=(10, 6))
    # Transpose for Time on X, Age on Y? Or heatmap standard?
    # Usually: plt.imshow(Z, aspect='auto', origin='lower')
    # Z[row, col]. Row is Y. Col is X.
    # We want Age on Y, Time on X.
    # rho_field is [Time, Age].
    # So Transpose.
    
    extent = [0, rho_field.shape[0]*0.01, r_grid[0], r_grid[-1]]
    
    plt.imshow(rho_field.T, aspect='auto', origin='lower', extent=extent, cmap='magma', vmin=0, vmax=50)
    plt.colorbar(label='Hazard Rate (Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Refractory Age (s)')
    plt.title('Empirical Hazard Field $\hat{\\rho}(r,t)$')
    plt.savefig(out_png)
    print(f"Saved {out_png}")

if __name__ == "__main__":
    plot_heatmap(sys.argv[1], sys.argv[2])
