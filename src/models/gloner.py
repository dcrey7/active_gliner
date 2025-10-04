"""
GLONER = GLiNER + LoRA

Simple helper to load GLiNER models with LoRA adapters.
Returns GLiNER models directly - use all GLiNER methods as normal.
"""

from gliner import GLiNER
from peft import get_peft_model, PeftModel, LoraConfig
import torch
from config.lora_defaults import DEFAULT_GLINER_MODEL, DEFAULT_LORA_CONFIG, DEFAULT_MAX_LENGTH


class GLONER:
    """
    Helper to load GLiNER models with LoRA adapters.

    Returns GLiNER models directly - no wrappers, just the model.
    Use all GLiNER methods directly: predict_entities(), run(), evaluate(), etc.
    """

    @staticmethod
    def default(logger):
        """
        Load default GLiNER model with default LoRA applied.

        Args:
            logger: Logger instance

        Returns:
            GLiNER model with LoRA applied (ready for training or inference)

        Example:
            model = GLONER.default(logger)
            entities = model.predict_entities(text, labels)
            train_lora_model(model, ...)
        """
        if logger:
            logger.info(f"Loading default GLONER: {DEFAULT_GLINER_MODEL}")

        # Load base GLiNER model
        model = GLiNER.from_pretrained(DEFAULT_GLINER_MODEL)
        model.config.max_len = DEFAULT_MAX_LENGTH

        if hasattr(model.data_processor, 'transformer_tokenizer'):
            model.data_processor.transformer_tokenizer.model_max_length = DEFAULT_MAX_LENGTH

        # Apply LoRA
        lora_config = LoraConfig(**DEFAULT_LORA_CONFIG)
        base_params = sum(p.numel() for p in model.model.parameters())

        if logger:
            logger.info(f"Applying LoRA: r={lora_config.r}, alpha={lora_config.lora_alpha}")

        model.model = get_peft_model(model.model, lora_config)
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

        if logger:
            logger.info(f"Trainable: {trainable_params:,} ({100*trainable_params/base_params:.1f}%)")
            logger.info("GLONER ready")

        return model

    @staticmethod
    def custom(logger, model_name=None, max_length=None, **lora_params):
        """
        Load custom GLiNER model with custom LoRA applied.

        Args:
            logger: Logger instance
            model_name: Optional custom GLiNER model (uses default if None)
            max_length: Optional custom max sequence length (uses default if None)
            **lora_params: Override any LoRA parameter:
                - r: LoRA rank
                - lora_alpha: LoRA alpha scaling
                - lora_dropout: Dropout probability
                - target_modules: List of modules to apply LoRA to
                - bias: Bias strategy ("none", "all", "lora_only")
                - task_type: PEFT task type

        Returns:
            GLiNER model with custom LoRA applied

        Examples:
            # Custom LoRA only
            model = GLONER.custom(logger, r=16, lora_alpha=32)

            # Custom model + LoRA
            model = GLONER.custom(
                logger,
                model_name="knowledgator/gliner-base",
                target_modules=["dense", "query"],
                r=16
            )

            # Custom max_length
            model = GLONER.custom(logger, max_length=4096)
        """
        model_name = model_name or DEFAULT_GLINER_MODEL
        max_length = max_length or DEFAULT_MAX_LENGTH

        if logger:
            logger.info(f"Loading custom GLONER: {model_name}")
            if lora_params:
                logger.info(f"Custom LoRA params: {list(lora_params.keys())}")

        # Load base GLiNER model
        model = GLiNER.from_pretrained(model_name)
        model.config.max_len = max_length

        if hasattr(model.data_processor, 'transformer_tokenizer'):
            model.data_processor.transformer_tokenizer.model_max_length = max_length

        # Apply LoRA with custom params
        lora_config_dict = DEFAULT_LORA_CONFIG.copy()
        lora_config_dict.update(lora_params)
        lora_config = LoraConfig(**lora_config_dict)

        base_params = sum(p.numel() for p in model.model.parameters())

        if logger:
            logger.info(f"Applying LoRA: r={lora_config.r}, alpha={lora_config.lora_alpha}")

        model.model = get_peft_model(model.model, lora_config)
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

        if logger:
            logger.info(f"Trainable: {trainable_params:,} ({100*trainable_params/base_params:.1f}%)")
            logger.info("GLONER ready")

        return model

    @staticmethod
    def load_with_adapter(adapter_path, logger, model_name=None, max_length=None):
        """
        Load GLiNER model with trained LoRA adapter for inference.

        Args:
            adapter_path: Path to saved LoRA adapter
            logger: Logger instance
            model_name: Optional custom GLiNER model (uses default if None)
            max_length: Optional custom max sequence length (uses default if None)

        Returns:
            GLiNER model with LoRA adapter loaded, in eval mode

        Example:
            model = GLONER.load_with_adapter("models/exp1/lora_adapter", logger)
            entities = model.predict_entities(text, labels)
            results = model.run(texts, labels)
        """
        model_name = model_name or DEFAULT_GLINER_MODEL
        max_length = max_length or DEFAULT_MAX_LENGTH

        if logger:
            logger.info(f"Loading {model_name} with adapter from {adapter_path}")

        # Load base GLiNER model
        model = GLiNER.from_pretrained(model_name)
        model.config.max_len = max_length

        if hasattr(model.data_processor, 'transformer_tokenizer'):
            model.data_processor.transformer_tokenizer.model_max_length = max_length

        # Load LoRA adapter
        model.model = PeftModel.from_pretrained(model.model, adapter_path)
        model.eval()

        # Move to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        if logger:
            logger.info(f"GLONER ready for inference on {device}")

        return model

    @staticmethod
    def get_param_counts(model):
        """
        Get parameter counts for a GLiNER model with LoRA.

        Args:
            model: GLiNER model with LoRA applied

        Returns:
            dict with 'total', 'trainable', 'percentage' keys
        """
        total = sum(p.numel() for p in model.model.parameters())
        trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        percentage = 100 * trainable / total if total > 0 else 0


        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,} , percentage: {percentage:.1f}%")
              
        return {
            'total': total,
            'trainable': trainable,
            'percentage': percentage
        }
