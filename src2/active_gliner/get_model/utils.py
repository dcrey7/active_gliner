import os
import logging
import gc
import torch
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import psutil
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

class SimpleTrainingMonitor(TrainerCallback):
    """Simple training monitor with resource tracking (no logger)"""

    def __init__(self, patience=10):
        self.patience = patience
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
            
            # Check for NaN
            if np.isnan(eval_loss) or np.isinf(eval_loss):
                logging.info("NaN validation loss detected! Stopping training.")
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
                logging.info(f"Step {state.global_step} | NEW BEST | "
                      f"Val Loss: {eval_loss:.4f} (improved {improvement:.4f}) | "
                      f"Train Loss: {current_train_loss:.4f} | "
                      f"Patience: {self.patience_counter}/{self.patience}")
            else:
                increase = eval_loss - self.best_loss
                self.patience_counter += 1
                logging.info(f"Step {state.global_step} | "
                      f"Val Loss: {eval_loss:.4f} (increased {increase:.4f}) | "
                      f"Train Loss: {current_train_loss:.4f} | "
                      f"Best: {self.best_loss:.4f} | "
                      f"Patience: {self.patience_counter}/{self.patience}")

            if self.patience_counter >= self.patience:
                logging.info("Early stopping triggered!")
                control.should_training_stop = True


def train_lora_model(model, train_data, eval_data, training_config, adapter_save_path):
    """
    Train a LoRA model and save the adapter weights (no logger).

    Args:
        model: GLiNER model with LoRA applied
        train_data: Training dataset
        eval_data: Evaluation dataset
        training_config: Dict with training parameters
        adapter_save_path: Path to save adapter
    """

    logging.info(f"Starting LoRA Training...")
    logging.info(f"Training samples: {len(train_data)}")
    logging.info(f"Eval samples: {len(eval_data)}")
    logging.info(f"Adapter will be saved to: {adapter_save_path}")
    
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
    
    # Create monitor
    monitor = SimpleTrainingMonitor(patience=training_config.get('patience', 3))
    
    # Create training arguments using the config
    training_args = TrainingArguments(
        save_strategy=training_config['save_strategy'],
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
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
        
        # Stability settings
        fp16=False,
        bf16=False,

        # Disable torch.compile - GLiNER has variable-length sequences that break it
        torch_compile=False,

        load_best_model_at_end=False,
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
    logging.info("Training started...")
    trainer.train()

    # Save the LoRA adapter weights
    logging.info(f"Saving LoRA adapter to: {adapter_save_path}")
    model.model.save_pretrained(adapter_save_path)

    logging.info("Training completed successfully!")

    # Generate training plots
    plot_training_metrics(monitor, adapter_save_path)

    # Cleanup trainer
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return True


def plot_training_metrics(monitor, adapter_save_path):
    """
    Create and save training metric plots.

    Args:
        monitor: SimpleTrainingMonitor instance with collected metrics
        adapter_save_path: Path where adapter is saved (plots will be in {adapter_save_path}/plots/)

    Saves:
        - training_curves.png: Loss, LR, and resource usage plots
        - training_summary.txt: Text summary of training
    """

    # Create plots directory
    plots_dir = os.path.join(adapter_save_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logging.info(f"\n📈 Generating training plots...")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # -------------------------------------------------------------------------
    # Plot 1: Training and Validation Loss
    # -------------------------------------------------------------------------
    if monitor.train_losses:
        axes[0].plot(monitor.steps, monitor.train_losses, 'b-', alpha=0.7,
                    label='Training Loss', linewidth=1.5)

    if monitor.eval_losses:
        axes[0].plot(monitor.eval_steps, monitor.eval_losses, 'r-', marker='o',
                    linewidth=2, markersize=6, label='Validation Loss')
        # Mark best loss
        best_idx = monitor.eval_losses.index(monitor.best_loss)
        axes[0].plot(monitor.eval_steps[best_idx], monitor.best_loss,
                    'g*', markersize=15, label=f'Best: {monitor.best_loss:.4f}')

    axes[0].set_xlabel('Steps', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Progress', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')  # Log scale for better visualization

    # -------------------------------------------------------------------------
    # Plot 2: Learning Rate Schedule
    # -------------------------------------------------------------------------
    if monitor.learning_rates:
        axes[1].plot(monitor.steps, monitor.learning_rates, 'g-', linewidth=2)
        axes[1].set_xlabel('Steps', fontsize=11)
        axes[1].set_ylabel('Learning Rate', fontsize=11)
        axes[1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # -------------------------------------------------------------------------
    # Plot 3: Resource Usage (GPU + CPU)
    # -------------------------------------------------------------------------
    if monitor.timestamps:
        ax3 = axes[2]

        # GPU memory (left y-axis)
        if monitor.gpu_memory:
            line1 = ax3.plot(monitor.timestamps, monitor.gpu_memory, 'purple',
                           linewidth=2, label='GPU Memory (GB)')
            ax3.set_xlabel('Time (minutes)', fontsize=11)
            ax3.set_ylabel('GPU Memory (GB)', color='purple', fontsize=11)
            ax3.tick_params(axis='y', labelcolor='purple')

        # CPU memory (right y-axis)
        if monitor.cpu_memory:
            ax3_twin = ax3.twinx()
            line2 = ax3_twin.plot(monitor.timestamps, monitor.cpu_memory, 'orange',
                                 linewidth=2, label='CPU Memory (%)')
            ax3_twin.set_ylabel('CPU Memory (%)', color='orange', fontsize=11)
            ax3_twin.tick_params(axis='y', labelcolor='orange')

            # Combined legend
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        ax3.set_title('Resource Usage Over Time', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Training plots saved to: {plot_path}")

    # -------------------------------------------------------------------------
    # Save Training Summary (Text File)
    # -------------------------------------------------------------------------
    summary_path = os.path.join(plots_dir, "training_summary.txt")

    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total steps: {len(monitor.steps)}\n")
        f.write(f"Total training time: {monitor.timestamps[-1]:.2f} minutes\n" if monitor.timestamps else "N/A\n")
        f.write(f"\n")

        f.write(f"Best validation loss: {monitor.best_loss:.4f}\n")
        if monitor.train_losses:
            f.write(f"Final training loss: {monitor.train_losses[-1]:.4f}\n")
        if monitor.eval_losses:
            f.write(f"Final validation loss: {monitor.eval_losses[-1]:.4f}\n")
        f.write(f"\n")

        if monitor.gpu_memory:
            f.write(f"Peak GPU memory: {max(monitor.gpu_memory):.2f} GB\n")
            f.write(f"Average GPU memory: {sum(monitor.gpu_memory)/len(monitor.gpu_memory):.2f} GB\n")

        if monitor.cpu_memory:
            f.write(f"Peak CPU memory: {max(monitor.cpu_memory):.1f}%\n")
            f.write(f"Average CPU memory: {sum(monitor.cpu_memory)/len(monitor.cpu_memory):.1f}%\n")

        f.write(f"\n")
        f.write(f"Early stopping: {'Yes' if monitor.patience_counter >= monitor.patience else 'No'}\n")
        if monitor.patience_counter >= monitor.patience:
            f.write(f"Stopped at step: {monitor.eval_steps[-1]}\n")

    logging.info(f"Training summary saved to: {summary_path}")

    return plots_dir
