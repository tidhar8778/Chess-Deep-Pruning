import pickle
import numpy as np

file = 'X.pkl'
file_labels = 'labels.pkl'
with open(file, 'rb') as f:
    data = pickle.load(f)

with open(file_labels, 'rb') as f:
    labels = pickle.load(f)

header = 'feature_1'
for d in range(2,391):
    header += ',feature_'+str(d)
header += ',label\n'



with open('data.csv', 'w') as f:
    f.write(header)
    for i, row in enumerate(data):
        for feat in row:
            f.write(str(feat) + ',')
        f.write(str(labels[i]) + '\n')

