"""Evaluation Module"""
from .evaluator import enhanced_evaluate
try:
    from .ner_evaluator import create_ner_evaluator
except:
    pass
