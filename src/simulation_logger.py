
import os
import sys
import json
import socket
import datetime
import subprocess
import platform
import uuid

class SimulationLogger:
    def __init__(self, script_name, config, out_file):
        """
        Initialize the logger.
        
        Args:
            script_name (str): Name of the script running (e.g., 'simulate_population.py').
            config (dict): The configuration dictionary.
            out_file (str): The primary output file path (e.g., 'results/run_X.npz').
        """
        self.script_name = script_name
        self.config = config
        self.out_file = out_file
        self.run_id = str(uuid.uuid4())
        self.timestamp = datetime.datetime.now().isoformat()
        
        # Determine output prefix for logs
        # E.g., results/run_X.npz -> results/logs/run_X
        out_abs = os.path.abspath(out_file)
        base_dir = os.path.dirname(out_abs)
        file_name = os.path.basename(out_abs)
        base_name = os.path.splitext(file_name)[0]
        
        # Standard Log Directory
        self.log_dir = os.path.join(base_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_prefix = os.path.join(self.log_dir, base_name)
        self.stdout_path = f"{self.log_prefix}.out"
        self.stderr_path = f"{self.log_prefix}.err"
        self.meta_path = f"{self.log_prefix}.meta.json"
        
        # Open Log Files
        self.stdout_f = open(self.stdout_path, 'w')
        self.stderr_f = open(self.stderr_path, 'w')
        
        # Redirect
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # We Tee to both console and file, or just file?
        # Requirement: "No silent failures". So we probably want to capture everything to file for audit.
        # But we also want to see progress on term.
        # Let's implement a Tee.
        sys.stdout = Tee(self.original_stdout, self.stdout_f)
        sys.stderr = Tee(self.original_stderr, self.stderr_f)
        
        self.write_metadata()
        
    def write_metadata(self):
        """
        Write comprehensive metadata to a JSON file.
        """
        # Git Hash
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        except:
            git_hash = "unknown"
            
        # CPU Count
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
        except:
            cpu_count = "unknown"
            
        meta = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "cpu_count": cpu_count,
            "script_name": self.script_name,
            "output_file": os.path.abspath(self.out_file),
            "git_commit": git_hash,
            "config": self.config
        }
        
        # Extract Key Physics Params for Searchability
        # (Assuming standard config structure)
        try:
            meta['N'] = self.config.get('population', {}).get('N')
            meta['J'] = self.config.get('coupling', {}).get('J')
            c = self.config.get('population', {}).get('shared_noise_fraction')
            meta['c'] = c
            meta['T'] = self.config.get('duration')
            meta['seed'] = self.config.get('defaults', {}).get('seed')
            
            pde = self.config.get('pde', {})
            meta['closure'] = 'jensen' if pde.get('use_jensen') else 'no_jensen'
            # Check for age_only implicitly via logic in simulation, but config might not say it explicit
            
        except Exception:
            pass
            
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=4)
            
    def close(self, exit_code=0):
        """
        Close logs and update metadata with exit status.
        """
        # Update metadata with exit code and end time
        try:
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
                
            meta['exit_code'] = exit_code
            meta['end_time'] = datetime.datetime.now().isoformat()
            
            with open(self.meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
        except:
            pass
            
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        self.stdout_f.close()
        self.stderr_f.close()

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
