import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from gliner import GLiNER
import torch
import os 

import random


#set seed
seed=42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)      

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add src path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src2')
sys.path.append(src_path)

#device
device=os.getenv("DEVICE","cuda" if torch.cuda.is_available() else "cpu")

print(device)
# Data loading
from active_gliner.create_data.gliner_format import convert_raw_json_to_gliner_training
from active_gliner.config.data_paths import (
    MIT_movies_NER_train_path,
    MIT_movies_NER_test_path,
    MIT_movies_NER_labels_path
)

with open(MIT_movies_NER_train_path, 'r') as f:
    train_data = json.load(f)

with open(MIT_movies_NER_test_path, 'r') as f:
    test_data = json.load(f)

with open(MIT_movies_NER_labels_path, 'r') as f:
    labels = json.load(f)

model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
model.to(device)

model.eval()
predictions=[]

for i in test_data[:5]:
    prediction=model.predict_entities((i['sentence']),labels,threshold=0.5,flat_ner=False)
    print(f"prediction is {prediction} \n")
    predictions.append(prediction)


# Convert test data to GLiNER format
converted_test_data = convert_raw_json_to_gliner_training(test_data)
for i in converted_test_data[:5]:
    print(i)

# Evaluate base model
print("\n" + "="*80)
print("BASE MODEL EVALUATION (No Fine-tuning)")
print("="*80)
results = model.evaluate(converted_test_data, entity_types=labels, batch_size=8, threshold=0.5,flat_ner=True)
print(results)


