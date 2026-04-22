import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def analyze_closure():
    # Load N=500 data
    data_path = 'results/diagnostics/bias_variance/sim_N500_s42.npz'
    if not os.path.exists(data_path):
        print("Data not found yet.")
        return

    data = np.load(data_path)
    if 'm_field' not in data:
        print("m_field not in data (Old simulation?).")
        return
        
    rho_field = data['rho_field'] # (Time, Age)
    m_field = data['m_field']     # (Time, Age)
    r_grid = data['r_grid']
    
    # Analyze Correlation
    # We expect rho approx exp(theta0 + theta2 * m) * exp(theta1 * V)? 
    # V is approximately constant (Mean Field 0).
    # So log(rho) ~ C + theta2 * m
    
    # theta_2 is -9800? That's huge.
    # m is small? a ranges 0..1?
    # Check scales.
    
    # Flatten
    # Filter out zeros (refractory) or low stats
    mask = rho_field > 0.1
    rho_valid = rho_field[mask]
    m_valid = m_field[mask]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(m_valid, np.log(rho_valid), alpha=0.1, s=1)
    
    # Fit
    # log_rho = slope * m + intercept
    A = np.vstack([m_valid, np.ones(len(m_valid))]).T
    m_fit, c_fit = np.linalg.lstsq(A, np.log(rho_valid), rcond=None)[0]
    
    plt.plot(m_valid, m_fit*m_valid + c_fit, 'r-', label=f'Fit: Slope={m_fit:.2f}')
    plt.xlabel('Mean Adaptation m(a, t)')
    plt.ylabel('Log Hazard log(rho(a, t))')
    plt.title('Closure Check: Hazard vs Adaptation')
    plt.legend()
    plt.savefig('results/diagnostics/closure_check.png')
    print("Saved closure plot.")
    print(f"Inferred Slope: {m_fit:.2f}")

if __name__ == "__main__":
    analyze_closure()
