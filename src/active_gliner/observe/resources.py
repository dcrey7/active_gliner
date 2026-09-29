"""Resource samples at training log steps."""

import os

import torch


def sample() -> dict:
    try:
        import psutil
    except ImportError:
        cpu = min(100.0, os.getloadavg()[0] / (os.cpu_count() or 1) * 100)
    else:
        cpu = psutil.cpu_percent()
    return {
        "gpu_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        "cpu_percent": cpu,
    }
