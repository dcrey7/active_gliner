"""
GLONER = GLiNER + LoRA

Clean wrapper for GLiNER models with LoRA support.
"""

from gliner import GLiNER
from peft import get_peft_model, PeftModel, LoraConfig
import torch
from typing import List, Dict, Union
from config.lora_defaults import DEFAULT_GLINER_MODEL, DEFAULT_LORA_CONFIG, DEFAULT_MAX_LENGTH
from data.transforms import prepare_texts_for_inference


class GLONER:
    """
    GLiNER model wrapper with LoRA support.

    Two factory methods:
    - for_training(): Creates trainable model with LoRA
    - for_inference(): Creates inference-only model with frozen adapter
    """

    def __init__(self, model):
        """Initialize with a GLiNER model"""
        self.model = model

    @staticmethod
    def for_training(base_model_path=None, lora_config=None, logger=None):
        """
        Create trainable GLONER with LoRA configuration.

        Args:
            base_model_path: Path to base GLiNER model (defaults to DEFAULT_GLINER_MODEL)
            lora_config: LoRA configuration dict (defaults to DEFAULT_LORA_CONFIG)
            logger: Optional logger instance

        Returns:
            GLONER instance with trainable LoRA
        """
        model_path = base_model_path or DEFAULT_GLINER_MODEL
        lora_cfg = lora_config or DEFAULT_LORA_CONFIG

        if logger:
            logger.info(f"Creating trainable GLONER from {model_path}")

        # Load base GLiNER model
        model = GLiNER.from_pretrained(model_path)
        model.config.max_len = DEFAULT_MAX_LENGTH

        if hasattr(model.data_processor, 'transformer_tokenizer'):
            model.data_processor.transformer_tokenizer.model_max_length = DEFAULT_MAX_LENGTH

        # Apply LoRA for training
        lora_config_obj = LoraConfig(**lora_cfg)
        model.model = get_peft_model(model.model, lora_config_obj)

        # # Set model to training mode
        # model.train()

        # Create GLONER wrapper and log parameter info
        gloner = GLONER(model)

        if logger:
            counts = gloner.get_param_counts()
            logger.info(f"LoRA applied: r={lora_config_obj.r}, alpha={lora_config_obj.lora_alpha}")
            logger.info(f"Trainable: {counts['trainable']:,} ({counts['percentage']:.1f}%)")

        return gloner

    @staticmethod
    def for_inference(base_model_path, adapter_path, logger=None):
        """
        Create inference GLONER with frozen adapter.

        Args:
            base_model_path: Path to base GLiNER model
            adapter_path: Path to trained LoRA adapter
            logger: Optional logger instance

        Returns:
            GLONER instance with frozen adapter for inference
        """

        base_model_path = base_model_path or DEFAULT_GLINER_MODEL
        if logger:
            logger.info(f"Creating inference GLONER from {base_model_path}")
            logger.info(f"Loading adapter: {adapter_path}")

        # Load base GLiNER model
        model = GLiNER.from_pretrained(base_model_path)
        model.config.max_len = DEFAULT_MAX_LENGTH

        if hasattr(model.data_processor, 'transformer_tokenizer'):
            model.data_processor.transformer_tokenizer.model_max_length = DEFAULT_MAX_LENGTH

        # Load trained adapter
        model.model = PeftModel.from_pretrained(model.model, adapter_path)

        # Set model to eval mode for inference
        model.eval()

        # Create GLONER wrapper and log parameter info
        gloner = GLONER(model)

        if logger:
            counts = gloner.get_param_counts()
            logger.info(f"Adapter loaded, model in eval mode")
            logger.info(f"Total params: {counts['total']:,}, Trainable: {counts['trainable']:,}")

        return gloner

    def get_param_counts(self, verbose: bool = False):
        """
        Get parameter counts for this model.

        Args:
            verbose: If True, print parameter counts

        Returns:
            Dict with 'total', 'trainable', 'percentage'
        """
        total = sum(p.numel() for p in self.model.model.parameters())
        trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        percentage = 100 * trainable / total if total > 0 else 0

        counts = {
            'total': total,
            'trainable': trainable,
            'percentage': percentage
        }

        if verbose:
            print(f"Total params: {counts['total']:,}")
            print(f"Trainable params: {counts['trainable']:,} ({counts['percentage']:.1f}%)")

        return counts

    def predict(self, data: Union[List[Dict], List[str], str], entity_types: List[str],
                threshold: float = 0.5, batch_size: int = 8, device: str = 'cpu', flat_ner: bool = True):
        """
        Predict entities in texts.

        Args:
            data: Can be:
                - List of NER format dicts with "tokenized_text" field
                - List of text strings
                - Single text string
            entity_types: List of entity types to extract
            threshold: Confidence threshold (0.0 to 1.0)
            batch_size: Batch size for inference
            device: Device to run inference on ('cpu' or 'cuda')
            flat_ner: Whether to use flat NER (no nested entities)

        Returns:
            Predictions in GLiNER format
        """
        # Prepare texts for inference
        if isinstance(data, str):
            # Single string
            texts = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # NER format data - use prepare function
            texts, _ = prepare_texts_for_inference(data)
        elif isinstance(data, list):
            # List of strings
            texts = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Run inference (model.run() handles device internally)
        with torch.no_grad():
            predictions = self.model.run(
                texts,
                entity_types,
                flat_ner=flat_ner,
                threshold=threshold,
                batch_size=batch_size
            )

        return predictions

    def to(self, device):
        """Move model to device"""
        self.model.to(device)
        return self

    def evaluate(self, data, entity_types, batch_size=8, threshold=0.5, flat_ner=True):
        """Evaluate model on data"""
        return self.model.evaluate(data, entity_types=entity_types, batch_size=batch_size, threshold=threshold, flat_ner=flat_ner)
    



