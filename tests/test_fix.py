
import subprocess
import os
import sys

def test_sim_exits():
    print("Testing simulate_population.py for infinite loop...")
    
    # create a dummy config
    with open('test_cfg_fix.yaml', 'w') as f:
        f.write("""
defaults:
  dt: 0.001
  seed: 42
population:
  N: 100
  shared_noise_fraction: 0.0
coupling:
  J: 0.0
  tau_s: 0.02
stimulus:
  mu_u: 0.0
  sigma_u: 0.0
  tau_c: 0.05
neuron:
  tau_m: 0.02
  tau_a: 0.2
  kappa: 0.0
  sigma: 0.0
spike_gen:
  refractory: 0.002
  theta_0: 1.0
  theta_1: 1.0
  theta_2: 0.0
  type: 'exponential'
duration: 0.1
save_fields: False
pde:
  R_max: 0.5
  dr: 0.005
  use_jensen: True
""")

    try:
        # Run with timeout. If it loops, it will timeout or print multiple lines.
        # We capture output to check for "Saved results" appearing multiple times.
        result = subprocess.run(
            ['python', 'src/simulate_population.py', '--config', 'test_cfg_fix.yaml', '--outfile', 'result_fix.npz'],
            capture_output=True,
            text=True,
            timeout=10 # Should take < 1s
        )
        
        output = result.stdout
        print("Output:", output)
        
        count = output.count("Saved results to")
        if count == 1:
            print("PASS: Saved exactly once.")
        elif count == 0:
            print("FAIL: Did not save.")
        else:
            print(f"FAIL: Saved {count} times (likely looping).")
            
    except subprocess.TimeoutExpired:
        print("FAIL: Timed out (likely looping).")
    except Exception as e:
        print(f"FAIL: Error {e}")
    finally:
        if os.path.exists('test_cfg_fix.yaml'): os.remove('test_cfg_fix.yaml')
        if os.path.exists('result_fix.npz'): os.remove('result_fix.npz')

if __name__ == "__main__":
    test_sim_exits()
