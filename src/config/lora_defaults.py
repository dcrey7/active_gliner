from peft import TaskType

DEFAULT_GLINER_MODEL = "knowledgator/modern-gliner-bi-large-v1.0"
DEFAULT_MAX_LENGTH = 8192

DEFAULT_LORA_CONFIG = {
    'r': 32,
    'lora_alpha': 64,
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
