import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def analyze_scaling(input_csv, out_csv, plot_path):
    df = pd.read_csv(input_csv)
    
    # Model: eps = A * N^-alpha
    # log(eps) = log(A) - alpha * log(N)
    
    metric = 'nrmse' if 'nrmse' in df.columns else 'rmse'
    print(f"Using metric: {metric}")
    
    x = np.log(df['N'])
    y = np.log(df[metric])
    
    # Linear fit
    p, cov = np.polyfit(x, y, 1, cov=True)
    alpha = -p[0]
    logA = p[1]
    
    # Uncertainty
    alpha_std = np.sqrt(cov[0,0])
    ci_95 = 1.96 * alpha_std
    
    print(f"Scaling Exponent alpha = {alpha:.4f} +/- {ci_95:.4f}")
    
    # Save stats
    stats = pd.DataFrame([{'alpha': alpha, 'alpha_ci_95': ci_95, 'A': np.exp(logA)}])
    stats.to_csv(out_csv, index=False)
    
    # Plot
    plt.figure()
    plt.loglog(df['N'], df[metric], 'ko', label='Data')
    
    N_fit = np.logspace(np.log10(df['N'].min()), np.log10(df['N'].max()), 100)
    eps_fit = np.exp(logA) * N_fit**(-alpha)
    
    plt.loglog(N_fit, eps_fit, 'r-', label=f'Fit $\\alpha={alpha:.2f} \pm {ci_95:.2f}$')
    
    # Ref 1/2
    eps_ref = eps_fit[0] * (N_fit/N_fit[0])**(-0.5)
    plt.loglog(N_fit, eps_ref, 'b:', label='Reference $N^{-1/2}$')
    
    plt.xlabel('N')
    plt.ylabel('RMSE')
    plt.title('Finite-Size Scaling (Validated)')
    plt.legend()
    plt.grid(True, which='both')
    plt.savefig(plot_path)
    print(f"Saved scaling plot to {plot_path}")

if __name__ == "__main__":
    import sys
    analyze_scaling(sys.argv[1], sys.argv[2], sys.argv[3])
