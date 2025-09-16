"""
Reproducibility utilities - simplified version of your set_all_seeds function
"""

import os
import random
import numpy as np
import torch
from typing import Optional
import logging


def set_all_seeds(seed=42, logger: Optional[logging.Logger] = None):
    """
    Set all random seeds exactly like your original function
    """
    if logger:
        logger.info(f"Setting all seeds to {seed} for reproducibility...")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (from your original code)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_torch_for_reproducibility():
    """Configure PyTorch settings from your original code"""
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True


def verify_seeds_work(seed=42):
    """Simple test to check if seed setting works"""
    # Test with the same seed twice
    random.seed(seed)
    r1 = random.random()
    
    random.seed(seed)
    r2 = random.random()
    
    return r1 == r2  # Should be True if seeds work
