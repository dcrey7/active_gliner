import json
with open('../results/high_mse_1000_examples.json', 'r') as file:
    low_n = json.load(file)

count=1
for i in low_n[-10:]:
    print((i["scores"]))
    count+=1

print(count)