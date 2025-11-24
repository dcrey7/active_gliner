from peft import TaskType


DEFAULT_GLINER_MODEL = "knowledgator/modern-gliner-bi-large-v1.0"
DEFAULT_MAX_LENGTH = 8192

DEFAULT_LORA_CONFIG = {
    'r':16 , # change to 64 in better gpu
    'lora_alpha': 32, # change to 128 in better gpu
    'lora_dropout': 0.1,
    'bias': 'none',
    'task_type': TaskType.TOKEN_CLS,
    'target_modules': [
        "dense", "projection", "Wqkv", "Wo", "Wi",
        "query", "key", "value",
        "intermediate.dense", "output.dense",
        "span_rep_layer.span_rep_layer.project_start.3",
        "span_rep_layer.span_rep_layer.project_start.0",
        "span_rep_layer.span_rep_layer.project_end.3",
        "span_rep_layer.span_rep_layer.project_end.0",
        "span_rep_layer.span_rep_layer.out_project.3",
        "span_rep_layer.span_rep_layer.out_project.0",
        'prompt_rep_layer.3',
        'prompt_rep_layer.0',
    ]
}

DEFAULT_TRAINING_CONFIG = {
        'num_steps': 2500,
        'save_strategy':"no",
        'train_batch_size': 8, # change to 12 for better gpu
        'gradient_accumulation_steps': 1,
        'learning_rate': 0.00022105770821309302,
        'others_lr': 5.8851860296580845e-06,
        'warmup_ratio': 0.15560507652730393,
        'eval_steps': 100,
        'save_steps': 100,
        'logging_steps': 10,
        'max_grad_norm': 1,
        'weight_decay': 0.08129586307822372,
        'others_weight_decay': 0.008521002768232644,
        'focal_loss_alpha': 0.75,
        'focal_loss_gamma': 1.0,
        'patience': 7

}