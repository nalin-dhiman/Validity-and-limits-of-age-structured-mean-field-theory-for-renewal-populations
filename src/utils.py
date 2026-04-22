import yaml
import numpy as np
import random
import os

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
