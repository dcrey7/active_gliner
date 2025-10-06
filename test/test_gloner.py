import sys
sys.path.append('../src')

from models.gloner import GLONER
from config.lora_defaults import DEFAULT_GLINER_MODEL, DEFAULT_LORA_CONFIG, DEFAULT_MAX_LENGTH
from utils.logging import get_logger
from models.gloner import GLONER
from data.loader import load_mit_dataset
from evaluation.enchanced_eval import enhanced_evaluate

logger = get_logger("GLONERTest")
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs visible: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


print(F"Default GLiNER model: {DEFAULT_GLINER_MODEL}")
print(f"Default GLiNER model max length: {DEFAULT_MAX_LENGTH}   ")
print("\nDefault LoRA config:")
for key, value in DEFAULT_LORA_CONFIG.items():
    if key == 'target_modules':
        print(f"  {key}: {len(value)} modules")
    else:
        print(f"  {key}: {value}")




# INFERENCE FLOW:




# Load GLiNER model with trained LoRA adapter
# model = GLONER.load_with_adapter("models/my_experiment/lora_adapter", logger)

test_data, entity_types = load_mit_dataset("../data/mit-movie/test.json", "../data/mit-movie/labels.json")  
print(test_data[0])
print(entity_types)
# OR with custom base model and max_length
# model = GLONER.load_with_adapter(
#     "../models/active_learning_adapter",
#     logger,
#     model_name="knowledgator/modern-gliner-bi-large-v1.0",
#     max_length=8192
# )
model=GLONER.default(logger)
print(model)
# Use all GLiNER methods directly
            # Enhanced evaluation on FULL test set
print(device)
model.to(device)
with torch.no_grad():
    gliner_results=model.evaluate(test_data, entity_types, batch_size=8, threshold=0.5 )

            
print(f"gliner_results : {gliner_results}")
# The model returned is just a GLiNER model - use it normally!

with torch.no_grad():
    llm_ft_results = enhanced_evaluate( model, test_data, entity_types,threshold=0.5, batch_size=8, has_ground_truth=True)            
print(f"llm_ft_results: {llm_ft_results['overall_metrics']}")