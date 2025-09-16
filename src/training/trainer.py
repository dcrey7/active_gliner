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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def intialize_model():
    """Initialize GLiNER model with LoRA configuration"""
    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
    model.config.max_len = 8192

    if hasattr(model.data_processor, 'transformer_tokenizer'):
        model.data_processor.transformer_tokenizer.model_max_length = 8192

    # Get base parameter count
    base_total = sum(p.numel() for p in model.model.parameters())
    print(f"Base Parameters: {base_total:,}")

    print("\n🔧 Applying LoRA Configuration...")

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
    print("✅ LoRA applied successfully!")

    # Get LoRA parameter count
    lora_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"📊 Trainable Parameters: {lora_trainable:,} ({100*lora_trainable/base_total:.1f}% of original)")

    return model


def load_evaluation_model(adapter_path, device='cuda'):
    """Load base model with LoRA adapter for evaluation"""
    # Load base model
    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
    model.config.max_len = 8192
    
    if hasattr(model.data_processor, 'transformer_tokenizer'):
        model.data_processor.transformer_tokenizer.model_max_length = 8192

    # Load LoRA adapter
    print(f"🔧 Loading LoRA adapters from {adapter_path}...")
    model.model = PeftModel.from_pretrained(model.model, adapter_path)
    model.eval()
    model.to(device)
    
    return model


def train_lora_model(model, train_data, eval_data, training_config, adapter_save_path):
    """
    Train a LoRA model and save the adapter weights
    
    Args:
        model: Pre-initialized GLiNER model with LoRA applied
        train_data: Training dataset (synthetic data)
        eval_data: Evaluation dataset (test data)
        training_config: Dictionary with training parameters
        adapter_save_path: Path to save the LoRA adapter weights
        
    Returns:
        bool: True if training completed successfully
    """
    
    print(f"🚀 Starting LoRA Training...")
    print(f"📊 Training samples: {len(train_data)}")
    print(f"📊 Eval samples: {len(eval_data)}")
    print(f"💾 Adapter will be saved to: {adapter_save_path}")
    
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
    )
    
    # Start training
    print("🔥 Training started...")
    train_result = trainer.train()
    
    # Save the LoRA adapter weights
    print(f"💾 Saving LoRA adapter to: {adapter_save_path}")
    model.model.save_pretrained(adapter_save_path)
    
    print("✅ Training completed successfully!")
    
    # Cleanup trainer
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return True