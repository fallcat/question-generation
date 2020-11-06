import torch
import nltk
import pickle

nltk.download('punkt')

roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
roberta.eval()

with open('data/cub/classes_w_descriptions_aab_ebird.tsv', 'rt') as input_file:
    data = [line.strip().split('\t') for line in input_file.readlines()]
    data = {line[0]: line[1] if len(line) == 2 else '' for line in data}

with open('data/cub/classes_w_descriptions_wiki.tsv', 'rt') as input_file:
    data2 = [line.strip().split('\t') for line in input_file.readlines()]
    for line in data2:
        if len(line) >= 2:
            data[line[0]] = ' '.join([data[line[0]], line[1]])

data_sents = {k: nltk.tokenize.sent_tokenize(v) for k, v in data.items()}
data_encoded = {k: [roberta.encode(s) for s in v] for k, v in data_sents.items()}
data_features = {k: [roberta.extract_features(s) for s in v] for k, v in data_encoded.items()}

with open('data/cub/descriptions_roberta.base.pkl', 'wb') as output_file:
    pickle.dump(data_features, output_file)