import sys
import os
import logging
import gc
import time
import torch
import warnings
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

print("=== Integration Test ===")
from utils.logging import setup_logging
from utils.reproducibility import set_all_seeds
from utils.device import setup_device

from config.settings import Settings
settings = Settings()

print(f"Settings cache_dir: {settings.cache_dir}")
print(f"Cache absolute path: {settings.cache_dir.resolve()}")
print(f"Does cache dir contain 'notebooks': {'notebooks' in str(settings.cache_dir)}")
settings





def intialize_model(device):



    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
    model.config.max_len = 8192

    if hasattr(model.data_processor, 'transformer_tokenizer'):    
        model.data_processor.transformer_tokenizer.model_max_length = 8192

    # Get base parameter count
    base_total = sum(p.numel() for p in model.model.parameters())
    print(f"Base Parameters: {base_total:,}")

    print("\n🔧 Applying FIXED LoRA Configuration...")

    # FIXED LoRA config - back to user's preferred values
    lora_config = LoraConfig(
        r=32,               # Back to 32 as requested
        lora_alpha=64,      # Back to 64 as requested
        target_modules=[
            # "query_proj", 
            # "key_proj",
            # "value_proj",
            "dense",
            "projection",
            "Wqkv", "Wo", "Wi",
            #   "linear_1", "linear_2",
            "query", "key", "value",  # BERT attention
        "intermediate.dense", "output.dense",  # BERT MLP,
        
        "span_rep_layer.span_rep_layer.project_start.3","span_rep_layer.span_rep_layer.project_start.0",
        "span_rep_layer.span_rep_layer.project_end.3","span_rep_layer.span_rep_layer.project_end.0",
        "span_rep_layer.span_rep_layer.out_project.3","span_rep_layer.span_rep_layer.out_project.0",
        'prompt_rep_layer.3','prompt_rep_layer.0',
        

        ],
        modules_to_save=[
                # "span_rep_layer",
            # "prompt_rep_layer"   # Only this one works properly
        ],
        lora_dropout=0.1,   # Reduced from 0.2
        bias="none",
        task_type=TaskType.TOKEN_CLS
    )

    # Apply LoRA
    model.model = get_peft_model(model.model, lora_config)

    # Manually make span_rep_layer trainable
    # for param in model.model.base_model.span_rep_layer.parameters():
    #     param.requires_grad = True

    print("✅ LoRA applied successfully!")

    # Get LoRA parameter count
    lora_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"📊 Trainable Parameters: {lora_trainable:,} ({100*lora_trainable/base_total:.1f}% of original)")

    model.to(device)

    print("Model after lora")
    
    return model

