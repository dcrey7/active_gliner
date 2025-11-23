# Active GLiNER

Information extraction from text using GLiNER and LLM distillation. This package enables cost-effective, locally-runnable named entity recognition by distilling LLM-extracted labels into lightweight GLiNER models.

## Overview

LLMs are powerful for information extraction but suffer from:
- High inference costs
- Cannot be fine-tuned easily
- Too large to run locally

Active GLiNER solves this by:
1. Using LLMs to generate high-quality training labels
2. Distilling those labels into fine-tuned GLiNER models
3. Mixing LLM labels with ground truth from business annotations
4. Producing lightweight, locally-runnable models with near-LLM performance

