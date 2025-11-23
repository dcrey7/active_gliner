from typing import List, Dict, Optional
import os
from gliner import GLiNER
from peft import PeftModel, LoraConfig, get_peft_model
# Import GLiNER training function
from .utils import train_lora_model
from .BluePrint import BluePrint
from ..config.model_configs import (
    DEFAULT_GLINER_MODEL,
    DEFAULT_MAX_LENGTH,
    DEFAULT_LORA_CONFIG,
    DEFAULT_TRAINING_CONFIG
)


class DefaultModel(BluePrint):
    """
    GLiNER model with default configuration.
    Implements BluePrint with GLiNER-specific logic.
    """

    def __init__(self, device=None, base_model_path=None, lora_config=None,
                 max_length=None, training_config=None, adapter_path=None):
        super().__init__(device)

        # Instance-level overrides
        self._override_base_path = base_model_path
        self._override_lora_config = lora_config
        self._override_max_length = max_length
        self._override_training_config = training_config
        self._override_adapter_path = adapter_path

    def get_base_model_path(self) -> str:
        return self._override_base_path or DEFAULT_GLINER_MODEL

    def get_lora_config(self) -> Optional[Dict]:
        if self._override_lora_config is not None:
            return self._override_lora_config
        return DEFAULT_LORA_CONFIG

    def get_max_length(self) -> int:
        return self._override_max_length or DEFAULT_MAX_LENGTH

    def get_training_config(self) -> Optional[Dict]:
        if self._override_training_config is not None:
            return self._override_training_config
        return DEFAULT_TRAINING_CONFIG

    def get_adapter_path(self) -> Optional[str]:
        return self._override_adapter_path

    def load_for_training(self, lora_config=None, max_length=None, base_model_path=None):
        """
        Load GLiNER model with training LoRA.
        Priority: method param > instance override > class default
        """
        # Resolve config
        base_path = base_model_path or self.get_base_model_path()
        lora_cfg = lora_config if lora_config is not None else self.get_lora_config()
        max_len = max_length or self.get_max_length()

        print(f"Loading GLiNER for training: {base_path}")

        # Load fresh GLiNER
        model = GLiNER.from_pretrained(base_path)

        # Configure tokenizer
        model.config.max_len = max_len
        if hasattr(model.data_processor, 'transformer_tokenizer'):
            model.data_processor.transformer_tokenizer.model_max_length = max_len

        # Apply LoRA
        if lora_cfg:
            print(f"Applying training LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
            config = LoraConfig(**lora_cfg)
            model.model = get_peft_model(model.model, config)
            model.model.print_trainable_parameters()
        else:
            print("No LoRA - training full model")

        self._model = model
        self._model.to(self.device)

        return self

    def load_for_inference(self, adapter_path=None, max_length=None, base_model_path=None):
        """
        Load GLiNER model optionally with adapter for inference.
        Priority: method param > instance override > class default
        """
        # Resolve config
        base_path = base_model_path or self.get_base_model_path()
        max_len = max_length or self.get_max_length()

        print(f"Loading GLiNER for inference: {base_path}")

        # Load fresh GLiNER
        model = GLiNER.from_pretrained(base_path)

        # Configure tokenizer
        model.config.max_len = max_len
        if hasattr(model.data_processor, 'transformer_tokenizer'):
            model.data_processor.transformer_tokenizer.model_max_length = max_len

        # Load adapter if provided and exists
        if adapter_path and self._adapter_exists(adapter_path):
            print(f"Loading adapter: {adapter_path}")
            model.model = PeftModel.from_pretrained(model.model, adapter_path)

            # Print active adapter info
            if hasattr(model.model, 'active_adapter'):
                print(f"Active adapter: {model.model.active_adapter}")
        else:
            if adapter_path:
                print(f"Adapter not found: {adapter_path}")
            print("Using base model")

        model.eval()
        self._model = model
        self._model.to(self.device)

        return self

    def fit(self, train_data, eval_data, adapter_save_path, training_config=None):
        """
        Train GLiNER with LoRA.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_for_training() first.")

        # Resolve training config
        train_cfg = training_config or self.get_training_config()
        if not train_cfg:
            raise RuntimeError("No training config provided")

  

        print(f"Training on {len(train_data)} examples")
        print(f"Eval on {len(eval_data)} examples")

        # Train
        train_lora_model(
            model=self._model,
            train_data=train_data,
            eval_data=eval_data,
            training_config=train_cfg,
            adapter_save_path=adapter_save_path
        )

        print(f"Adapter saved to {adapter_save_path}")

        return self

    def predict_entities(self, text, entity_types, threshold=0.5, flat_ner=False):
        """
        Predict entities using GLiNER.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_for_inference() first.")

        return self._model.predict_entities(text, entity_types, threshold=threshold, flat_ner=flat_ner)

    def _adapter_exists(self, adapter_path: str) -> bool:
        """Check if adapter files exist."""
        if not os.path.exists(adapter_path):
            return False

        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        return all(os.path.exists(os.path.join(adapter_path, f)) for f in required_files)
