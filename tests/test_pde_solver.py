import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from age_pde import AgePDESolver

def test_mass_conservation():
    dt = 0.0001
    R_max = 0.5
    dr = 0.001
    
    solver = AgePDESolver(None, dt, R_max, dr, ref_period=0.002)
    
    # Run for 100 steps with constant hazard
    rho = 10.0 # 10 Hz
    
    for i in range(100):
        solver.step(rho)
        mass = solver.get_mass()
        assert abs(mass - 1.0) < 5e-4, f"Mass conservation violation at step {i}: {mass}"

def test_nonnegativity():
    dt = 0.0001
    R_max = 0.5
    dr = 0.001
    solver = AgePDESolver(None, dt, R_max, dr, ref_period=0.002)
    
    rho = 50.0 
    for i in range(100):
        solver.step(rho)
        assert np.all(solver.q >= -1e-10), "Negative density detected"

if __name__ == "__main__":
    test_mass_conservation()
    test_nonnegativity()
    print("Tests passed.")
