import torch
import nltk
import pickle
from tqdm import tqdm
import numpy as np

nltk.download('punkt')

roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
roberta.eval()

with open('data/cub/classes_w_descriptions_aab_ebird.tsv', 'rt') as input_file:
    data = [line.strip().split('\t') for line in input_file.readlines()]
    keys = [line[0] for line in data]
    data = {line[0]: line[1] if len(line) == 2 else '' for line in data}

with open('data/cub/classes_w_descriptions_wiki.tsv', 'rt') as input_file:
    data2 = [line.strip().split('\t') for line in input_file.readlines()]
    for line in data2:
        if len(line) >= 2:
            data[line[0]] = ' '.join([data[line[0]], line[1]])

data_sents = {k: nltk.tokenize.sent_tokenize(v) for k, v in data.items()}
data_encoded = {k: [roberta.encode(s) for s in v] for k, v in data_sents.items()}
data_features = {}
for k, v in tqdm(data_encoded.items()):
    data_features[k] = [roberta.extract_features(s).detach().numpy() for s in v]

flattened_features = []
flattened_features_keys = []
sentences = []
for i, key in enumerate(keys):
    flattened_features.extend([d for d in data_features[key]])
    flattened_features_keys.extend([i] * len(data_features[key]))
    sentences.extend([d for d in data_sents[key]])


with open('data/cub/descriptions_roberta.base.pkl', 'wb') as output_file:
    pickle.dump({'sentences': sentences,
                 'data_features': data_features,
                 'flattened_features': flattened_features,
                 'flattened_features_keys': flattened_features_keys},
                output_file)