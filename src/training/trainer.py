"""
Training module for LoRA fine-tuning experiments
Contains all model and training related functions
"""
import os
import gc
import torch
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from utils.logging import get_logger  # ← Import for fallback only
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from data.transforms import convert_synthetic_to_ner_format, validate_and_clean_ner_data
from config.settings import Settings
import psutil
import time
import pandas as np
import numpy as np

# Remove global settings and logger - these will be passed from main
# settings = Settings()
# settings.setup()
# logger = get_logger("ActiveLearning")  # ← REMOVED
# set_all_seeds(seed=settings.global_seed, logger=logger)
# device = setup_device(logger=logger)


class SimpleTrainingMonitor(TrainerCallback):
    """Simple training monitor with resource tracking"""
    
    def __init__(self, patience=10, logger=None):
        self.logger = logger if logger else get_logger("ActiveLearning")  # Fallback
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.steps = []
        self.eval_steps = []
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Resource tracking
        self.gpu_memory = []
        self.cpu_memory = []
        self.timestamps = []
        self.start_time = time.time()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
                
                # Track resources
                current_time = (time.time() - self.start_time) / 60  # minutes
                self.timestamps.append(current_time)
                
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                    self.gpu_memory.append(gpu_mem)
                else:
                    self.gpu_memory.append(0.0)
                
                cpu_mem = psutil.virtual_memory().percent
                self.cpu_memory.append(cpu_mem)
                
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:  # Every 50 steps
            torch.cuda.empty_cache()
            gc.collect()
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_loss' in metrics:
            eval_loss = metrics['eval_loss']
            
            # Check for NaN - CRITICAL FIX
            if np.isnan(eval_loss) or np.isinf(eval_loss):
                self.logger.info(f"NaN validation loss detected! Stopping training.")
                control.should_training_stop = True
                return
                
            self.eval_losses.append(eval_loss)
            self.eval_steps.append(state.global_step)
            
            # Get current training loss for comparison
            current_train_loss = self.train_losses[-1] if self.train_losses else 0.0
            
            # Log train and validation loss
            self.logger.info(f"Step {state.global_step}: Train Loss = {current_train_loss:.4f}, Val Loss = {eval_loss:.4f}")
            
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.patience_counter = 0
                self.logger.info(f"New best validation loss: {eval_loss:.4f}")
            else:
                self.patience_counter += 1
                self.logger.info(f"Validation loss: {eval_loss:.4f} | Patience: {self.patience_counter}/{self.patience}")
                
            if self.patience_counter >= self.patience:
                self.logger.info("Early stopping triggered!")
                control.should_training_stop = True

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:  # Every 50 steps
            torch.cuda.empty_cache()
            gc.collect()
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_loss' in metrics:
            eval_loss = metrics['eval_loss']
            
            # Check for NaN - CRITICAL FIX
            if np.isnan(eval_loss) or np.isinf(eval_loss):
                self.logger.info(f"🚨 NaN validation loss detected! Stopping training.")
                control.should_training_stop = True
                return
                
            self.eval_losses.append(eval_loss)
            self.eval_steps.append(state.global_step)
            
            # Get current training loss for comparison
            current_train_loss = self.train_losses[-1] if self.train_losses else 0.0
            
            if eval_loss < self.best_loss:
                improvement = self.best_loss - eval_loss
                self.best_loss = eval_loss
                self.patience_counter = 0
                self.logger.info(f"🎯 Step {state.global_step} | NEW BEST! | "
                               f"Val Loss: {eval_loss:.4f} (↓{improvement:.4f}) | "
                               f"Train Loss: {current_train_loss:.4f} | "
                               f"Patience: {self.patience_counter}/{self.patience}")
            else:
                increase = eval_loss - self.best_loss
                self.patience_counter += 1
                patience_emoji = "⚠️" if self.patience_counter >= self.patience - 1 else "📈"
                self.logger.info(f"{patience_emoji} Step {state.global_step} | "
                               f"Val Loss: {eval_loss:.4f} (↑{increase:.4f}) | "
                               f"Train Loss: {current_train_loss:.4f} | "
                               f"Best: {self.best_loss:.4f} | "
                               f"Patience: {self.patience_counter}/{self.patience}")
                
            if self.patience_counter >= self.patience:
                self.logger.info("🛑 Early stopping triggered!")
                control.should_training_stop = True
    
def intialize_model(logger=None):
    """Initialize GLiNER model with LoRA configuration"""
    if logger is None:
        logger = get_logger("ActiveLearning")  # Fallback
    
    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
    model.config.max_len = 8192

    if hasattr(model.data_processor, 'transformer_tokenizer'):
        model.data_processor.transformer_tokenizer.model_max_length = 8192

    # Get base parameter count
    base_total = sum(p.numel() for p in model.model.parameters())
    logger.info(f"Base Parameters: {base_total:,}")

    logger.info("Applying LoRA Configuration...")

    # LoRA config
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "dense", "projection", "Wqkv", "Wo", "Wi",
            "query", "key", "value",
            "intermediate.dense", "output.dense",
            "span_rep_layer.span_rep_layer.project_start.3",
            "span_rep_layer.span_rep_layer.project_start.0",
            "span_rep_layer.span_rep_layer.project_end.3",
            "span_rep_layer.span_rep_layer.project_end.0",
            "span_rep_layer.span_rep_layer.out_project.3",
            "span_rep_layer.span_rep_layer.out_project.0",
            'prompt_rep_layer.3','prompt_rep_layer.0',
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.TOKEN_CLS
    )

    # Apply LoRA
    model.model = get_peft_model(model.model, lora_config)
    logger.info("LoRA applied successfully!")

    # Get LoRA parameter count
    lora_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {lora_trainable:,} ({100*lora_trainable/base_total:.1f}% of original)")

    return model


def load_evaluation_model(adapter_path, device='cuda', logger=None):
    """Load base model with LoRA adapter for evaluation"""
    if logger is None:
        logger = get_logger("ActiveLearning")  # Fallback
    
    # Load base model
    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
    model.config.max_len = 8192
    
    if hasattr(model.data_processor, 'transformer_tokenizer'):
        model.data_processor.transformer_tokenizer.model_max_length = 8192

    # Load LoRA adapter
    logger.info(f"Loading LoRA adapters from {adapter_path}...")
    model.model = PeftModel.from_pretrained(model.model, adapter_path)
    model.eval()
    model.to(device)
    
    return model


def train_lora_model(model, train_data, eval_data, training_config, adapter_save_path, logger=None):
    """
    Train a LoRA model and save the adapter weights
    
    Args:
        model: Pre-initialized GLiNER model with LoRA applied
        train_data: Training dataset (synthetic data)
        eval_data: Evaluation dataset (test data)
        training_config: Dictionary with training parameters
        adapter_save_path: Path to save the LoRA adapter weights
        logger: Logger instance (required for proper logging)
        
    Returns:
        bool: True if training completed successfully
    """
    if logger is None:
        logger = get_logger("ActiveLearning")  # Fallback
    
    logger.info(f"Starting LoRA Training...")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Eval samples: {len(eval_data)}")
    logger.info(f"Adapter will be saved to: {adapter_save_path}")
    
    # Create output directory for training checkpoints
    checkpoint_dir = os.path.join(os.path.dirname(adapter_save_path), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(adapter_save_path, exist_ok=True)
    
    # Setup data collator
    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True
    )
    
    # Create monitor with the logger
    monitor = SimpleTrainingMonitor(patience=training_config['patience'], logger=logger)
    
    # Create training arguments using the config
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,  # For training checkpoints
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.020216630535603918),
        others_lr=training_config['others_lr'],
        others_weight_decay=training_config.get('others_weight_decay', 0.020216630535603918),
        lr_scheduler_type=training_config.get('lr_scheduler_type', "cosine"),
        warmup_ratio=training_config['warmup_ratio'],
        per_device_train_batch_size=training_config['train_batch_size'],
        per_device_eval_batch_size=training_config['train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        max_steps=training_config['num_steps'],
        max_grad_norm=training_config['max_grad_norm'],
        
        # Focal loss parameters
        focal_loss_alpha=training_config.get('focal_loss_alpha', 0.75),
        focal_loss_gamma=training_config.get('focal_loss_gamma', 1.0),
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=training_config['eval_steps'],
        save_steps=training_config['save_steps'],
        save_total_limit=3,
        logging_steps=training_config['logging_steps'],
        seed=42,
        dataloader_num_workers=0,
        use_cpu=False,
        report_to="none",
        
        # Stability settings
        fp16=False,
        bf16=False,
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
        callbacks=[monitor]
    )
    
    # Start training
    logger.info("Training started...")
    train_result = trainer.train()
    
    # Save the LoRA adapter weights
    logger.info(f"Saving LoRA adapter to: {adapter_save_path}")
    model.model.save_pretrained(adapter_save_path)
    
    logger.info("Training completed successfully!")
    
    # Cleanup trainer
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return True