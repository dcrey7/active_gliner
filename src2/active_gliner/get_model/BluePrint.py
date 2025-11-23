from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch
import os
import gc
from peft import PeftModel


class BluePrint(ABC):
    """
    Model-agnostic base class for NER models with LoRA support.
    """

    def __init__(self, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None

    @abstractmethod
    def get_base_model_path(self) -> str:
        """Return model path."""
        pass

    @abstractmethod
    def get_lora_config(self) -> Optional[Dict]:
        """Return LoRA config or None."""
        pass

    @abstractmethod
    def get_max_length(self) -> int:
        """Return max sequence length."""
        pass

    @abstractmethod
    def get_training_config(self) -> Optional[Dict]:
        """Return training config."""
        pass

    @abstractmethod
    def get_adapter_path(self) -> Optional[str]:
        """Return default adapter path or None."""
        pass

    @abstractmethod
    def load_for_training(self, lora_config=None, max_length=None, base_model_path=None):
        """Load model for training."""
        pass

    @abstractmethod
    def load_for_inference(self, adapter_path=None, max_length=None, base_model_path=None):
        """Load model for inference."""
        pass

    @abstractmethod
    def fit(self, train_data, eval_data, adapter_save_path, training_config=None):
        """Train the model."""
        pass

    @abstractmethod
    def predict_entities(self, text, entity_types, threshold=0.5, flat_ner=False):
        """Predict entities."""
        pass

    def to_cpu(self):
        """Move model to CPU and free GPU memory."""
        if self._model is not None:
            self._model.to('cpu')
            self.device = 'cpu'
            torch.cuda.empty_cache()

    def to_gpu(self, device='cuda'):
        """Move model to GPU."""
        if self._model is not None:
            self._model.to(device)
            self.device = device

    def cleanup_memory(self):
        """Clean up model and free memory."""
        if self._model is not None:
            self.to_cpu()
            del self._model
            self._model = None
            gc.collect()
            torch.cuda.empty_cache()

    def get_trainable_parameters(self) -> Dict:
        """Get trainable parameter counts."""
        if self._model is None:
            return {'trainable': 0, 'total': 0, 'percentage': 0.0}

        # Check if using PEFT
        if isinstance(self._model.model, PeftModel):
            # Use PEFT native method
            trainable, total = self._model.model.get_nb_trainable_parameters()
            return {
                'trainable': trainable,
                'total': total,
                'percentage': (trainable / total * 100) if total > 0 else 0.0
            }
        else:
            # Manual count for non-PEFT models
            total = sum(p.numel() for p in self._model.model.parameters())
            trainable = sum(p.numel() for p in self._model.model.parameters() if p.requires_grad)
            return {
                'trainable': trainable,
                'total': total,
                'percentage': (trainable / total * 100) if total > 0 else 0.0
            }

    def print_trainable_parameters(self):
        """Print trainable parameter info."""
        if self._model is None:
            print("Model not loaded")
            return

        if isinstance(self._model.model, PeftModel):
            # Use PEFT native print
            self._model.model.print_trainable_parameters()
        else:
            # Manual print
            params = self.get_trainable_parameters()
            print(f"trainable params: {params['trainable']:,} || "
                  f"all params: {params['total']:,} || "
                  f"trainable%: {params['percentage']:.4f}")

    def get_info(self) -> Dict:
        """Get model information."""
        info = {
            'model_class': self.__class__.__name__,
            'base_model_path': self.get_base_model_path(),
            'max_length': self.get_max_length(),
            'lora_config': self.get_lora_config(),
            'training_config': self.get_training_config(),
            'adapter_path': self.get_adapter_path(),
            'device': self.device,
        }

        if self._model is not None:
            info['trainable_parameters'] = self.get_trainable_parameters()

        return info
