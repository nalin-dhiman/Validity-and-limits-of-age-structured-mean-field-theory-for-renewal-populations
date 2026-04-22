
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import subprocess
import multiprocessing
import traceback
import csv
import time
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import load_config, ensure_dir
from metrics import ActivityMetrics
import json

class Status:
    OK = "OK"
    BREAKDOWN_NAN = "BREAKDOWN_NAN" # NaN/Inf
    BREAKDOWN_GUARD = "BREAKDOWN_GUARD" # Deprecated generic
    BREAKDOWN_MASS = "BREAKDOWN_MASS" # Hard Mass
    BREAKDOWN_TRUNCATION = "BREAKDOWN_TRUNCATION" # Hard Tail
    BREAKDOWN_DIVERGENCE = "BREAKDOWN_DIVERGENCE" 
    MISMATCH_HIGH_EPS = "MISMATCH_HIGH_EPS" 
    MISMATCH_TRUNCATION = "MISMATCH_TRUNCATION" # Soft Tail
    SOFT_MASS_LEAK = "SOFT_MASS_LEAK" # Soft Mass
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA" 
    PIPELINE_ERROR = "PIPELINE_ERROR"
    IO_ERROR = "IO_ERROR"
    TIMEOUT = "TIMEOUT"

def run_point(args):
    """
    Executes a single simulation point.
    Returns a dictionary of result metadata/status.
    """
    cfg, J, c, seed, out_dir = args
    fname = f'sim_J{J:.3f}_c{c:.1f}_s{seed}'
    
    # Robustness Tag Support
    if 'robustness_tag' in cfg:
        fname += f"_{cfg['robustness_tag']}"
    
    # Result Record
    record = {
        'J': J, 'c': c, 'seed': seed,
        'status': Status.PIPELINE_ERROR, 
        'reason': '',
        'nan_flag': False,
        'guard_flag': False,
        'diverged_flag': False,
        'exit_code': -1,
        'runtime_s': 0.0,
        'output_path': '',
        'error_msg': '',
        # New Diagnostics
        'mass_min': np.nan,
        'mass_final': np.nan,
        'tail_mass_max': np.nan,
        'cum_leak_final': np.nan,
        't_fail': np.nan,
        'R_max': np.nan
    }
    
    # Configure
    run_cfg = cfg.copy()
    run_cfg['coupling']['J'] = float(J)
    run_cfg['population']['shared_noise_fraction'] = float(c)
    run_cfg['defaults']['seed'] = int(seed)
    
    # T=60s for stability scanning, unless smoke mode
    if cfg.get('smoke_mode', False):
        run_cfg['duration'] = 2.0
    else:
        run_cfg['duration'] = 60.0
        
    run_cfg['population']['N'] = 5000 
    run_cfg['pde']['use_jensen'] = True
    
    tmp_cfg = os.path.join(out_dir, f'{fname}.yaml')
    out_npz = os.path.join(out_dir, f'{fname}.npz')
    record['output_path'] = out_npz
    
    # Write Config
    try:
        with open(tmp_cfg, 'w') as f:
            yaml.dump(run_cfg, f)
    except Exception as e:
        record['status'] = Status.PIPELINE_ERROR
        record['error_msg'] = f"Config write failed: {e}"
        return record
        
    # Check Exists (if reusing, though we usually clean)
    if os.path.exists(out_npz):
        # We assume if it exists it might be stale unless checked. 
        # For simplicity in this hardened pipeline, we allow overwrite.
        pass

    t0 = time.time()
    try:
        print(f"[PhaseScan] Running J={J:.3f} c={c:.1f} s={seed}...")
        
        # Decide Script: PDE-only or Full
        script_name = 'src/simulate_population.py'
        extra_args = ['--online-pde']
        
        if cfg.get('pde_only', False):
             script_name = 'src/simulate_pde.py'
             extra_args = [] # simulate_pde doesn't need --online-pde flag
             print("... (PDE ONLY Mode) ...")
             
        cmd = ['python', script_name, '--config', tmp_cfg, '--outfile', out_npz] + extra_args
        
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        record['exit_code'] = res.returncode
        record['runtime_s'] = time.time() - t0
        
    except subprocess.CalledProcessError as e:
        record['exit_code'] = e.returncode
        record['runtime_s'] = time.time() - t0
        
        # Classify Failure from STDERR/STDOUT
        err_out = e.stderr + e.stdout
        
        if "NaN" in err_out or "Inf" in err_out:
            record['status'] = Status.BREAKDOWN_NAN
            record['reason'] = "NaN in simulation"
            record['nan_flag'] = True
        elif "GUARD: [Mass] (HARD)" in err_out or "PDE BREAKDOWN: Mass" in err_out:
            record['status'] = Status.BREAKDOWN_MASS
            record['reason'] = "Mass Guard Triggered"
            record['guard_flag'] = True
        elif "GUARD: [TailMass] (HARD)" in err_out or "PDE BREAKDOWN: Tail" in err_out:
             record['status'] = Status.BREAKDOWN_TRUNCATION
             record['reason'] = "Truncation Breakdown (TailMass)"
             record['guard_flag'] = True
        elif "divergence" in err_out or "PDE BREAKDOWN: Divergence" in err_out:
             record['status'] = Status.BREAKDOWN_DIVERGENCE
             record['reason'] = "Divergence Detected"
             record['diverged_flag'] = True
        elif "Positivity" in err_out or "CFL" in err_out:
             record['status'] = Status.BREAKDOWN_GUARD
             record['reason'] = "Stability Guard"
             record['guard_flag'] = True
        else:
            record['status'] = Status.PIPELINE_ERROR
            record['reason'] = "Subprocess Failed"
            record['error_msg'] = "Subprocess Failed"
        
        # Save Error Log
        log_dir = os.path.join(os.path.dirname(out_dir), 'logs')
        ensure_dir(log_dir)
        err_file = os.path.join(log_dir, f'{fname}.err')
        with open(err_file, 'w') as f:
            f.write(f"Command: {' '.join(e.cmd)}\n")
            f.write(f"Exit Code: {e.returncode}\n\n")
            f.write(">>> STDERR:\n")
            f.write(e.stderr)
            f.write("\n>>> STDOUT:\n")
            f.write(e.stdout)
            
        return record
        
    except Exception as e:
        record['status'] = Status.PIPELINE_ERROR
        record['error_msg'] = f"Python Exception: {e}"
        return record

    # Post-Run Validation
    if not os.path.exists(out_npz):
        record['status'] = Status.IO_ERROR
        record['error_msg'] = "Output file missing"
        return record
        
    try:
        # Lightweight Validation
        data = np.load(out_npz, allow_pickle=True)
        
        # Check integrity
        if 'A' not in data:
            raise ValueError("Key 'A' missing in NPZ")
            
        A = data['A']
        
        # 1. NaN Check
        if np.isnan(A).any() or np.isinf(A).any():
            record['status'] = Status.BREAKDOWN_NAN
            record['reason'] = "NaN/Inf in Output"
            record['nan_flag'] = True
            return record
            
        # 2. Divergence Check (A > A_max)
        if np.max(A) > 200.0:
            record['status'] = Status.BREAKDOWN_DIVERGENCE
            record['reason'] = f"Rate Divergence (Max={np.max(A):.1f})"
            record['diverged_flag'] = True
            return record
            
        # 3. Truncation / Mass Check from Metrics
        # 3. Truncation / Mass Check from Metrics
        if 'pde_metrics' in data:
            pm = data['pde_metrics'].item()
            record['mass_min'] = pm.get('mass_min', np.nan)
            record['mass_final'] = pm.get('mass_final', np.nan)
            record['tail_mass_max'] = pm.get('tail_mass_max', np.nan)
            record['cum_leak_final'] = pm.get('cum_leak_final', np.nan)
            record['t_fail'] = pm.get('t_fail', np.nan)
            record['R_max'] = pm.get('R_max', np.nan)
            
            # Logic for Soft Guards
            # Only apply if Status is currently OK (don't override Hard Breakdowns provided by exit code logic if they passed exit code but wrote data?)
            # Usually Hard Breakdowns exit non-zero and are handled in except CaughtProcessError.
            # If we are here, exit code was 0.
            
            if record['tail_mass_max'] > 1e-4:
                 record['status'] = Status.MISMATCH_TRUNCATION
                 record['reason'] = f"Soft Truncation (Tail={record['tail_mass_max']:.1e})"
                 return record
                 
            if record['mass_min'] < 0.99:
                 record['status'] = Status.SOFT_MASS_LEAK
                 record['reason'] = f"Soft Mass Leak (Min={record['mass_min']:.4f})"
                 return record
                 
        # 4. Mass Array Check (Legacy)
        if 'mass' in data:
            mass = data['mass']
            if np.isnan(mass).any():
                record['status'] = Status.BREAKDOWN_NAN
                record['reason'] = "NaN in Mass"
                record['nan_flag'] = True
                return record
                  
        record['status'] = Status.OK
        record['reason'] = "Success"
        
    except Exception as e:
        record['status'] = Status.IO_ERROR # Corrupted file
        record['error_msg'] = f"Validation Failed: {e}"
        
    return record

def run_scan(config_path, out_dir, args):
    # 1. Setup
    ensure_dir(out_dir)
    log_dir = os.path.join(os.path.dirname(out_dir), 'logs')
    ensure_dir(log_dir)
    
    # Clean if fresh start (unless rerun mode?)
    # For this task, we assume full sweep.
    if args.clean and os.path.exists(out_dir):
        print(f"Cleaning {out_dir}...")
        for f in os.listdir(out_dir):
            path = os.path.join(out_dir, f)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
             
    cfg = load_config(config_path)
    if args.smoke:
        cfg['smoke_mode'] = True
        print("[PhaseScan] SMOKE MODE ENABLED: Using tiny grid (2x2, 1 seed)")
        Js = np.array([0.0, 2.0])
        cs = [0.0, 0.5]
        seeds = [0]
    else:
        # Standard Grid
        Js = np.linspace(0.0, 2.0, 11) # Step 0.2
        cs = [0.0, 0.2, 0.5, 0.8]
        seeds = [0, 1, 2] # 3 seeds per condition
        
    # ZOOM MODE Override
    if args.zoom:
        print(f"[PhaseScan] ZOOM MODE: J=[{args.J_min}, {args.J_max}] dJ={args.dJ}")
        # Parse c_list
        try:
            cs = [float(x) for x in args.c_list.split(',')]
        except:
            print(f"Invalid c-list: {args.c_list}. Using default [0.0, 0.5].")
            cs = [0.0, 0.5]
            
        Js = np.arange(args.J_min, args.J_max + 1e-9, args.dJ)
        # Seeds passed via args?
        # User said "same seed protocol" but user command example showed --seeds 0,1,2
        if args.seeds:
             seeds = [int(x) for x in args.seeds.split(',')]
        else:
             seeds = [0, 1, 2]

    # ROBUSTNESS TRIPLET Override
    if args.robustness_triplet:
        print("[PhaseScan] ROBUSTNESS TRIPLET MODE")
        # Triplet: Stable (J=0, c=0), Borderline (J=1.0, c=0.0 approx?), Breakdown (J=2.0, c=0.5)
        # Actually user said: "Stable reference, Borderline point, Hard breakdown point"
        # We need to define these.
        # Let's pick fixed representative points:
        # 1. Stable: J=0.0, c=0.0
        # 2. Borderline: J=1.2, c=0.2 (Typical transition)
        # 3. Breakdown: J=2.0, c=0.8
        
        triplets = [
            (0.0, 0.0, "Stable"),
            (1.2, 0.2, "Borderline"),
            (2.0, 0.8, "Breakdown")
        ]
        
        # Sweeps: dt_scale {1.0, 0.5}, tau {20ms, 50ms} -> actually tau is post-process, dt is sim
        # "sweep dt_scale"
        dt_base = cfg['defaults']['dt']
        dt_scales = [1.0, 0.5]
        
        # We need to run simulations for these (J, c) * dt_scales * seeds
        # We will modify tasks generation to include dt in config override?
        # But our run_point doesn't support changing dt easily without config hack.
        # Let's adjust run_point or just hack it here.
        # run_point takes (cfg, J, c, seed, out_dir).
        # We can pass a modified cfg to run_point!
        
        tasks = []
        seeds = [0] # 1 seed enough for robustness check? User said "overlay trajectories", usually 1 seed.
        # "For each point, sweep dt_scale... Output classification stability table..."
        
        # We'll just execute them.
        # But run_point usage below expects (cfg, J, c, seed, out_dir).
        # We can pre-multiply tasks with different CFGs.
        
        for J, c, label in triplets:
            for dts in dt_scales:
                r_cfg = cfg.copy()
                r_cfg['defaults']['dt'] = dt_base * dts
                # Label in filename? run_point uses generic fname.
                # We need to hack fname generation in run_point or out_dir.
                # Let's output to subfolders or use distinct J/c/s combos?
                # J/c are fixed.
                # Maybe use seed to encode dt? No, that's confusing.
                # Let's make separate calls to a new runner or just hack run_point to use 'robustness_tag' in cfg?
                
                r_cfg['robustness_tag'] = f"dt{dts}"
                
                # We need run_point to respect this tag for filename.
                tasks.append((r_cfg, J, c, 0, out_dir))
                
        print(f"Robustness Triplet Tasks: {len(tasks)}")
        
        # Execute differently? No, use standard executor but need to patch run_point for filename.
        # See PATCH below.
    else:            
        # Standard Tasks
        pass # Handle below
        
    # Status CSV init
    status_csv = os.path.join(out_dir, 'status.csv')
    fieldnames = ['J', 'c', 'seed', 'status', 'reason', 'nan_flag', 'guard_flag', 'diverged_flag', 
                  'exit_code', 'runtime_s', 'output_path', 'error_msg', 
                  'mass_min', 'mass_final', 'tail_mass_max', 'cum_leak_final', 't_fail', 'R_max']
    
    if not os.path.exists(status_csv):
        with open(status_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
    # Prepare Tasks
    tasks = []
    
    # If rerun list provided? (Not in this step, but good for future)
    if args.rerun_list:
        if os.path.exists(args.rerun_list):
            print(f"[PhaseScan] RERUN MODE: Loading tasks from {args.rerun_list}")
            rerun_df = pd.read_csv(args.rerun_list)
            for _, row in rerun_df.iterrows():
                tasks.append((cfg, row['J'], row['c'], row['seed'], out_dir))
        else:
            print(f"Rerun list {args.rerun_list} not found.")
            return
    else:
        # Full Grid / Zoom / Robustness
        if args.robustness_triplet:
             # Tasks already generated above
             pass
        else:
             for c in cs:
                for J in Js:
                    for s in seeds:
                        tasks.append((cfg, J, c, s, out_dir))
                
    print(f"Total Phase Scan Simulations: {len(tasks)}")
    
    if not tasks:
        print("No tasks to run.")
        post_process_scan(out_dir, cfg, cs)
        return

    # 2. Golden Run (Sequential 1st task) - Only if not rerun or if rerun includes it?
    # If rerun, we might skip golden check for speed, but safer to keep it if list is short.
    # Let's just run tasks.
    
    print(">>> executing GOLDEN RUN (1st task) <<<")
    golden_res = run_point(tasks[0])
    
    # Log result
    log_result(status_csv, golden_res, fieldnames)
    
    if golden_res['status'] != Status.OK:
        print("\n" + ("!"*60))
        print(f"CRITICAL FAILURE: GOLDEN RUN FAILED with {golden_res['status']}")
        print(f"Error: {golden_res['error_msg']}")
        print(("!"*60) + "\n")
        sys.exit(1)
    else:
        print(">>> Golden Run SUCCESS <<<")
        
    # 3. Parallel Execution
    remaining = tasks[1:]
    if remaining:
        max_cores = multiprocessing.cpu_count()
        n_proc = min(max(1, int(max_cores * 0.8)), 32)
        if args.smoke: n_proc = 1 # Sequential for smoke
        
        print(f"Running scan with {n_proc} processes...")
        
        with multiprocessing.Pool(processes=n_proc) as pool:
            # We use imap_unordered to log as we go? 
            # Or just map. Map is fine, but we want incremental logging.
            # Using imap allows us to write to CSV as results arrive.
            
            for res in pool.imap_unordered(run_point, remaining):
                log_result(status_csv, res, fieldnames)
                if res['status'] != Status.OK:
                    print(f"Failure: J={res['J']:.2f} c={res['c']:.1f} s={res['seed']} -> {res['status']}")

    # 4. Generate Rerun List & Analyze
    post_process_scan(out_dir, cfg, cs, args.smoke)

def log_result(csv_path, record, fieldnames):
    # Lock? CSV append is usually atomic-ish for single lines, but multiprocess might interleave.
    # actually run_point is in worker, log_result is in main process if using imap. YES.
    # So no lock needed if main process writes.
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(record)

def check_coverage(df, cs, smoke_mode=False):
    """
    Enforce Phase IX Coverage Standards.
    """
    if smoke_mode:
        print("[PhaseScan] SMOKE MODE: Skipping coverage checks.")
        return True

    # 1. Global Coverage (Valid Statuses)
    # 1. Global Coverage (Valid Statuses)
    valid_statuses = [
        Status.OK, 
        Status.MISMATCH_HIGH_EPS,
        Status.MISMATCH_TRUNCATION,
        Status.SOFT_MASS_LEAK,
        Status.BREAKDOWN_NAN, 
        Status.BREAKDOWN_MASS,
        Status.BREAKDOWN_TRUNCATION,
        Status.BREAKDOWN_GUARD, 
        Status.BREAKDOWN_DIVERGENCE
    ]
    
    total = len(df)
    valid = df['status'].isin(valid_statuses).sum()
    coverage = valid / total if total > 0 else 0.0
    
    print(f"[Coverage] {valid}/{total} points valid ({coverage*100:.1f}%)")
    
    if coverage < 0.95:
        print("!!! CRITICAL FAILURE: Coverage < 95% !!!")
        print("Identify pipeline errors and rerun.")
        return False
        
    # 2. Seed Sufficiency
    # Group by (J, c) and count valid seeds
    # valid_mask = df['status'].isin(valid_statuses) # Already calculated effectively
    # But we need to check per point
    df_valid = df[df['status'].isin(valid_statuses)]
    counts = df_valid.groupby(['J', 'c']).size()
    
    insufficient = counts[counts < 2]
    if not insufficient.empty:
        print(f"!!! CRITICAL FAILURE: {len(insufficient)} points have < 2 valid seeds !!!")
        print(insufficient)
        return False
        
    return True

def post_process_scan(out_dir, cfg, cs, smoke_mode=False):
    print("\n[PhaseScan] Post-processing results...")
    status_file = os.path.join(out_dir, 'status.csv')
    if not os.path.exists(status_file):
        print("No status file found.")
        return
        
    df = pd.read_csv(status_file)
    
    # 1. Rerun List (Failures)
    # Failures are anything NOT in valid set used for science
    # valid_statuses for science: OK, MISMATCH, BREAKDOWN_* 
    # So Rerun = PIPELINE_ERROR, IO_ERROR, TIMEOUT, or Missing
    
    valid_sci_statuses = [
        Status.OK, 
        Status.MISMATCH_HIGH_EPS,
        Status.MISMATCH_TRUNCATION,
        Status.SOFT_MASS_LEAK,
        Status.BREAKDOWN_NAN, 
        Status.BREAKDOWN_MASS,
        Status.BREAKDOWN_TRUNCATION,
        Status.BREAKDOWN_GUARD, 
        Status.BREAKDOWN_DIVERGENCE
    ]
    
    rerun_mask = ~df['status'].isin(valid_sci_statuses)
    rerun_df = df[rerun_mask]
    
    rerun_list_path = os.path.join(out_dir, 'rerun_list.csv')
    if not rerun_df.empty:
        rerun_df.to_csv(rerun_list_path, index=False)
        print(f"Generated {rerun_list_path} with {len(rerun_df)} entries.")
    else:
        if os.path.exists(rerun_list_path): os.remove(rerun_list_path)
        print("No failures requiring rerun found.")
        
    # 2. Coverage Check
    if not check_coverage(df, cs, smoke_mode):
        print("Coverage gating failed. Skipping boundary extraction.")
        # We generally Exit Non-Zero here in a real pipeline, but let's just return
        if not smoke_mode:
            sys.exit(100) # Distinct exit code for Coverage Failure
        return

    # 3. Analysis & Figures
    analyze_results(df, out_dir, cfg, cs)

def analyze_results(df_status, out_dir, cfg, cs):
    # Filter for Valid or Breakdown
    # BREAKDOWN counts as valid data point with infinite error
    
    # DEDUPLICATE: Keep last entry for each (J, c, seed)
    df_unique = df_status.sort_values('runtime_s').drop_duplicates(subset=['J', 'c', 'seed'], keep='last')
    

    valid_mask = df_unique['status'].isin([
        Status.OK, 
        Status.MISMATCH_HIGH_EPS,
        Status.MISMATCH_TRUNCATION,
        Status.SOFT_MASS_LEAK,
        Status.BREAKDOWN_NAN, 
        Status.BREAKDOWN_MASS,
        Status.BREAKDOWN_TRUNCATION,
        Status.BREAKDOWN_GUARD, 
        Status.BREAKDOWN_DIVERGENCE
    ])
    df_valid = df_unique[valid_mask].copy()
    
    # Generate Heatmap CSVs
    # 1. Breakdown Type Grid
    # 2. Coverage Grid (Seeds count)
    
    # Breakdown Priority: NAN > GUARD > DIVERGENCE > MISMATCH > OK
    # We want to map (J, c) to a dominant status
    
    print(f"Analyzing {len(df_valid)} valid/breakdown points...")
    results = []
    dt = cfg['defaults']['dt']
    tau = 0.05
    burn_steps = int(10.0 / dt)
    
    print(f"Analyzing {len(df_valid)} valid/breakdown points...")
    
    for _, row in df_valid.iterrows():
        J, c, s = row['J'], row['c'], row['seed']
        status = row['status']
        
        if status.startswith("BREAKDOWN"):
            # Infinite Error
            results.append({
                'J': J, 'c': c, 'Seed': s,
                'NRMSE': np.inf, 'RMSE': np.inf,
                'Status': status
            })
            continue
            
        # Process OK files
        path = row['output_path']
        if not os.path.exists(path):
            # Should have been caught, but double check
            continue
            
        try:
            data = np.load(path, allow_pickle=True)
            A_mc = data['A'][burn_steps:]
            if 'A_pde_adapt' in data:
                A_pde = data['A_pde_adapt'][burn_steps:]
            else:
                A_pde = data['A_pde'][burn_steps:]
            
            L = min(len(A_mc), len(A_pde))
            A_mc = A_mc[:L]
            A_pde = A_pde[:L]
            
            A_mc_sm = ActivityMetrics.smooth_trace(A_mc, dt, tau)
            A_pde_sm = ActivityMetrics.smooth_trace(A_pde, dt, tau)
            
            rmse = np.sqrt(np.mean((A_mc_sm - A_pde_sm)**2))
            norm = np.std(A_mc_sm)
            nrmse = rmse / norm if norm > 1e-9 else 0.0
            
            results.append({
                'J': J, 'c': c, 'Seed': s,
                'NRMSE': nrmse, 'RMSE': rmse,
                'Status': status
            })
            
            # Disk Cleanup
            # os.remove(path) # Optional: keep for debugging in this phase? 
            # User said "disk cleanup" in previous script. I will keep it but maybe only for successful ones.
            os.remove(path)
            
        except Exception as e:
            print(f"Error analyzing {path}: {e}")
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_dir, 'phase_scan_metrics.csv'), index=False)
    
    # Save Truncation Metrics
    if 'tail_mass_max' in df_valid.columns:
        trunc_cols = ['J', 'c', 'seed', 'mass_min', 'mass_final', 'tail_mass_max', 'cum_leak_final', 't_fail', 'R_max', 'status']
        # Filter columns that actually exist (in case of old data or partial run)
        valid_cols = [c for c in trunc_cols if c in df_valid.columns]
        df_valid[valid_cols].to_csv(os.path.join(out_dir, 'truncation_metrics.csv'), index=False)
        print("Saved truncation_metrics.csv")
    
    if df_res.empty:
        return
        
    # Boundary Extraction
    extract_boundary(df_res, out_dir, cs)
    
def extract_boundary(df, out_dir, cs):
    # Robust Baseline: Median + k*MAD
    # Pivot to find baseline dist
    baseline_df = df[(df['J'] == 0.0) & (df['c'] == 0.0)]
    if baseline_df.empty:
        min_J = df['J'].min()
        baseline_df = df[df['J'] == min_J]
        
    vals = baseline_df['NRMSE'].values
    vals = vals[np.isfinite(vals)] # Filter Infs if J=0 has breakdown (unlikely)
    
    if len(vals) == 0:
        eps_base = 0.0
        mad = 0.01
    else:
        eps_base = np.median(vals)
        mad = np.median(np.abs(vals - eps_base))
    
    # k=5 for robustness
    k = 5.0
    eps_tol = eps_base + k * mad
    if mad < 1e-5: eps_tol = eps_base + 0.1 # Fallback
    
    print(f"Boundary Logic: Median={eps_base:.4f}, MAD={mad:.4f}, Threshold={eps_tol:.4f}")
    
    # Boundary Finding
    # Group by J, c
    # We require minimum coverage?
    # For now, simplest logic: Mean NRMSE of seeds > Threshold -> Instability
    # Handling INF: if any seed is INF -> Mean is INF -> Break.
    
    df_agg = df.groupby(['J', 'c'])['NRMSE'].agg(['median', 'max']).reset_index()
    
    boundary = []
    for c in cs:
        sub = df_agg[df_agg['c'] == c].sort_values('J')
        found = False
        for _, row in sub.iterrows():
            # If Median > Tol OR Max is Inf (Breakdown)
            if row['median'] > eps_tol or np.isinf(row['max']):
                 boundary.append({'c': c, 'J_star': row['J'], 'J_err': 0.05, 'NRMSE_break': row['median']})
                 found = True
                 break
        if not found:
             boundary.append({'c': c, 'J_star': 2.0, 'J_err': 0.0, 'NRMSE_break': np.nan})
             
    pd.DataFrame(boundary).to_csv(os.path.join(out_dir, 'phase_boundary.csv'), index=False)
    print("Saved boundary.")
    
    pd.DataFrame(boundary).to_csv(os.path.join(out_dir, 'phase_boundary.csv'), index=False)
    print("Saved boundary.")
    
    # Save Heatmap Data
    results_df = pd.DataFrame(df)
    results_df.to_csv(os.path.join(out_dir, 'eps_grid.csv'), index=False)
    
    # Breakdown Type Grid
    # Pivot status
    status_pivot = results_df.pivot_table(index='c', columns='J', values='Status', aggfunc='first') # Just take one if multiple seeds, or mode?
    status_pivot.to_csv(os.path.join(out_dir, 'breakdown_type_grid.csv'))
    
    # 4. Coverage Heatmap (Implied by valid counts in df_agg or similar)
    # We can compute seed count per point
    seed_counts = results_df.groupby(['J', 'c']).size().reset_index(name='count')
    seed_counts.to_csv(os.path.join(out_dir, 'coverage_heatmap.csv'), index=False)
    
    # Plot Coverage Heatmap
    plot_heatmaps(results_df, out_dir, cs)

def plot_heatmaps(df, out_dir, cs):
    # 1. Coverage/Status Heatmap (Implied by availability)
    pass # Todo if needed, but metrics csv + per-seed status is good.
    
    # Plot Boundary
    # Re-plot standard figure
    boundary_df = pd.read_csv(os.path.join(out_dir, 'phase_boundary.csv'))
    
    plt.figure(figsize=(8,6))
    plt.plot(boundary_df['c'], boundary_df['J_star'], 'o-', linewidth=2)
    plt.xlabel('Shared Noise c')
    plt.ylabel('Critical Coupling J*')
    plt.title('Phase Boundary (Robust)')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Fig_PhaseBoundary.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--out_dir', default='results/phase_boundary')
    parser.add_argument('--smoke', action='store_true', help='Run tiny smoke test grid')
    parser.add_argument('--clean', action='store_true', help='Clean output dir before start')
    parser.add_argument('--rerun-list', default=None, help='CSV file with failed points to rerun')
    parser.add_argument('--zoom', action='store_true', help='Enable zoom scan mode')
    parser.add_argument('--J-min', type=float, default=0.0)
    parser.add_argument('--J-max', type=float, default=2.0)
    parser.add_argument('--dJ', type=float, default=0.05)
    parser.add_argument('--c-list', type=str, default="0.0,0.5")
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--pde-only', action='store_true', help='Run PDE reference only (no coupling feedback)')
    parser.add_argument('--robustness_triplet', action='store_true', help='Run targeted robustness triplet sweep')
    args = parser.parse_args()
    
    run_scan(args.config, args.out_dir, args)
