"""
Helper functions for LLM Inference
Converts LLM predictions to GLiNER format for evaluation
"""

from typing import List, Dict, Any


def convert_ner_to_gliner_format(ner_predictions: List[Dict]) -> List[List[Dict]]:
    """
    Convert NER predictions to GLiNER prediction format
    
    NER format: {"tokenized_text": [...], "ner": [[start, end, type], ...]}
    GLiNER format: [[{"start": char_pos, "end": char_pos, "label": type, "text": str, "score": float}], ...]
    
    Args:
        ner_predictions: List of NER format predictions
        
    Returns:
        GLiNER format predictions (list of lists of dicts)
    """
    gliner_predictions = []
    
    for example in ner_predictions:
        tokenized_text = example['tokenized_text']
        ner_spans = example['ner']
        
        example_predictions = []
        
        for start_token, end_token, entity_type in ner_spans:
            # Convert token positions to character positions
            char_start = 0
            for i in range(start_token):
                char_start += len(tokenized_text[i]) + 1  # +1 for space
            
            char_end = char_start
            for i in range(start_token, end_token + 1):
                char_end += len(tokenized_text[i])
                if i < end_token:
                    char_end += 1  # +1 for space
            
            # Get entity text
            entity_text = " ".join(tokenized_text[start_token:end_token+1])
            
            # Create GLiNER format prediction
            example_predictions.append({
                'start': char_start,
                'end': char_end,
                'label': entity_type,
                'text': entity_text,
                'score': 1.0  # LLM doesn't provide confidence scores
            })
        
        gliner_predictions.append(example_predictions)
    
    return gliner_predictions


class LLMGLiNERWrapper:
    """
    Wraps pre-computed predictions to look like a GLiNER model
    Allows using enhanced_evaluate() with LLM predictions
    
    Usage:
        llm_results = llm_inference.generate(...)
        gliner_format = convert_ner_to_gliner_format(llm_results['all_labels'])
        mock_model = LLMGLiNERWrapper(gliner_format)
        results = enhanced_evaluate(mock_model, test_data, entity_types)
    """
    
    def __init__(self, predictions: List[List[Dict]]):
        """
        Initialize mock model with pre-computed predictions
        
        Args:
            predictions: GLiNER format predictions
        """
        self.predictions = predictions
        self.model_name = "LLM (Mock)"
    
    def batch_predict_entities(
        self,
        texts: List[str],
        labels: List[str],
        threshold: float = 0.5,
        **kwargs
    ) -> List[List[Dict]]:
        """
        LLM-backed prediction method (returns pre-computed predictions)
        
        Args:
            texts: Input texts (ignored, we use pre-computed)
            labels: Entity labels (ignored)
            threshold: Confidence threshold (ignored, LLM has no scores)
            
        Returns:
            Pre-computed predictions
        """
        return self.predictions[:len(texts)]
    
    def to(self, device):
        """LLM-backed to() method for device placement"""
        return self
    
    def eval(self):
        """LLM-backed eval() method"""
        return self


def create_llm_gliner_wrapper(ner_predictions: List[Dict]) -> LLMGLiNERWrapper:
    """
    Convenience function to convert NER predictions and wrap in mock model
    
    Args:
        ner_predictions: NER format predictions
        
    Returns:
        LLMGLiNERWrapper instance ready for enhanced_evaluate()
    """
    gliner_format = convert_ner_to_gliner_format(ner_predictions)
    return LLMGLiNERWrapper(gliner_format)
