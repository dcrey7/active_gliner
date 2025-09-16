import json
with open('../results/low_score_1000_examples.json', 'r') as file:
    low_n = json.load(file)

print(len(low_n[:5]))