"""
Memory management utilities - simplified version of your cleanup_memory function
"""

import gc
import psutil
import torch
from typing import Optional
import logging


def cleanup_memory(logger: Optional[logging.Logger] = None):
    """
    Clean up memory exactly like your original function
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    if logger:
        logger.info("Memory cleanup completed")


def get_memory_info():
    """Get current memory usage info"""
    info = {}
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        info['gpu_allocated_gb'] = gpu_allocated
        info['gpu_reserved_gb'] = gpu_reserved
    else:
        info['gpu_allocated_gb'] = 0
        info['gpu_reserved_gb'] = 0
    
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    info['cpu_used_gb'] = cpu_memory.used / 1024**3
    info['cpu_percent'] = cpu_memory.percent
    info['cpu_available_gb'] = cpu_memory.available / 1024**3
    
    return info


def log_memory_status(logger: logging.Logger):
    """Log current memory status"""
    info = get_memory_info()
    logger.info(f"Memory Status:")
    logger.info(f"  GPU: {info['gpu_allocated_gb']:.1f}GB allocated, {info['gpu_reserved_gb']:.1f}GB reserved")
    logger.info(f"  CPU: {info['cpu_used_gb']:.1f}GB used ({info['cpu_percent']:.1f}%)")
