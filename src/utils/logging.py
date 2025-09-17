"""
Logging utilities - simplified version of your original setup_logging function
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir="logs", logger_name="ActiveLearning"):
    """
    Setup logging exactly like your original function but cleaner
    """
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = Path(log_dir) / f"{logger_name}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(logger_name)
    logger.info("="*80)
    logger.info("ACTIVE LEARNING PIPELINE WITH PROPER TRAIN/TEST SEPARATION")
    logger.info("="*80)
    logger.info(f"Log file: {log_filename}")
    
    return logger


def get_logger(name="ActiveLearning"):
    """Get existing logger"""
    return logging.getLogger(name)
