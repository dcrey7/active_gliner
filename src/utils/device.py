"""
Device setup utilities - simplified version of your original device setup code
"""

import torch
from typing import Optional
import logging


def setup_device(logger: Optional[logging.Logger] = None):
    """
    Setup device exactly like your original code
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if logger:
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            log_cuda_info(logger)
    
    return device


def log_cuda_info(logger: logging.Logger):
    """Log CUDA information like your original code"""
    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return
    
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Number of GPUs visible: {torch.cuda.device_count()}")
    logger.info(f"Current GPU: {torch.cuda.current_device()}")
    
    if torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU Name: {props.name}")
        logger.info(f"GPU Memory: {props.total_memory / 1024**3:.1f} GB")


def get_device_info():
    """Get basic device information"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        info['gpu_name'] = props.name
        info['gpu_memory_gb'] = props.total_memory / 1024**3
        info['current_device'] = torch.cuda.current_device()
    
    return info
