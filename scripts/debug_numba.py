import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from numba_kernels import step_pde_numba
import time

def test_numba_pde():
    print("Testing Numba PDE Kernel...")
    
    # Setup
    N_r = 65000
    q = np.random.random(N_r)
    rho_r = np.random.random(N_r) * 10.0
    dt = 0.0001
    dr = 0.001
    A_t = 10.0
    q_out = np.zeros_like(q)
    
    start = time.time()
    # Call 1 (Compile)
    step_pde_numba(q, rho_r, dt, dr, A_t, q_out)
    print(f"Compilation + Run 1: {time.time()-start:.4f}s")
    
    start = time.time()
    # Call 2 (Run)
    for i in range(1000):
        step_pde_numba(q, rho_r, dt, dr, A_t, q_out)
        # Swap logic
        q[:] = q_out[:] 
        
    print(f"1000 Steps: {time.time()-start:.4f}s")
    print("Success.")

if __name__ == "__main__":
    test_numba_pde()
