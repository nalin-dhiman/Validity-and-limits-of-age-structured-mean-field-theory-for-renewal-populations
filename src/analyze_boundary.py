import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def analyze_boundary(input_csv, out_csv, plot_path, baseline_ref=None):
    df = pd.read_csv(input_csv)
    
    metric = 'nrmse' if 'nrmse' in df.columns else 'rmse'
    print(f"Using metric: {metric}")
    
    # 1. Determine Baseline Error (Uncoupled)
    # J=0, c=0
    baseline_row = df[(df['J'] == 0.0) & (df['c'] == 0.0)]
    if baseline_row.empty:
        raise ValueError("No baseline (J=0, c=0) found.")
    
    eps_0 = baseline_row[metric].iloc[0]
    eps_tol = 2.0 * eps_0
    
    print(f"Baseline Error eps_0: {eps_0:.4f}")
    print(f"Tolerance eps_tol: {eps_tol:.4f}")
    
    # 2. Extract Boundary J*(c)
    # For each c, find smallest J where error > eps_tol
    # Note: J_vals are discrete. 
    # If all J fit, J* is > max(J).
    # If J=0 fails (impossible by def), J*=0.
    
    c_vals = sorted(df['c'].unique())
    boundary = []
    
    for c in c_vals:
        sub = df[df['c'] == c].sort_values('J')
        # Find first J where error > eps_tol
        exceed = sub[sub[metric] > eps_tol]
        if not exceed.empty:
            J_star = exceed['J'].iloc[0]
            # Refine? Linear interp?
            # Discrete J is sparse. Let's just report the discrete value for now or midpoint.
            # Let's reported the first failing J.
        else:
            # valid for all J tested
            J_star = sub['J'].max() # Or 'valid'
            
        boundary.append({'c': c, 'J_star': J_star, 'valid': exceed.empty})
        
    b_df = pd.DataFrame(boundary)
    b_df.to_csv(out_csv, index=False)
    print(b_df)
    
    # 3. Enhanced Plot
    # Grid
    pivot = df.pivot(index='J', columns='c', values=metric)
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis',
                cbar_kws={'label': metric.upper()})
                
    # Overlay Boundary
    # Map J_star to y-axis coordinates
    # Y-axis is index of J.
    # X-axis is index of c.
    
    # Need indices
    j_u = sorted(df['J'].unique())
    c_u = sorted(df['c'].unique())
    j_map = {v: i for i, v in enumerate(j_u)}
    c_map = {v: i for i, v in enumerate(c_u)}
    
    y_coords = []
    x_coords = []
    
    for idx, row in b_df.iterrows():
        c_val = row['c']
        j_val = row['J_star']
        if row['valid']:
            # All valid, no boundary crossing within range
             continue
        
        x_coords.append(c_map[c_val] + 0.5)
        y_coords.append(j_map[j_val] + 0.5)
        
    if x_coords:
        plt.plot(x_coords, y_coords, 'r*-', markersize=15, linewidth=3, label='Boundary $J^*$')
        
    plt.title(f'Phase Diagram (Tol = {eps_tol:.2f})')
    plt.gca().invert_yaxis() # Heatmap 0 at top usually, check sns
    # sns heatmap 0 at top? yes.
    # J is sorted 0..max. 
    # If yticklabels are 0..max, then 0 is at bottom? 
    # sns puts row 0 at top. row 0 is J=0.
    # We want J=0 at bottom for cartesian intuition?
    # Usually heatmap (0,0) is top-left.
    # Let's explicitly set invert_yaxis to put J=0 at bottom.
    plt.gca().invert_yaxis()
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved refined boundary plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('out_csv')
    parser.add_argument('plot_path')
    args = parser.parse_args()
    
    analyze_boundary(args.input_csv, args.out_csv, args.plot_path)
