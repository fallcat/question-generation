import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance

import torch
import spacy
import nltk
import benepar
from benepar.spacy_plugin import BeneparComponent

max_k = 25

benepar.download('benepar_en')
nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent('benepar_en'))

nltk.download('punkt')

roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
roberta.eval()


# doc = nlp("The time for action is now. It's never too late to do something.")
# sent = list(doc.sents)[0]
# print(sent._.parse_string)
# # (S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))
# print(sent._.labels)
# # ('S',)
# print(list(sent._.children)[0])
# for c in sent._.children:
#     print(list(c._.children))

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

sentences = []
sentences_keys = []
for i, key in enumerate(keys):
    sentences.extend([d for d in data_sents[key]])
    sentences_keys.extend([i] * len(data_sents[key]))

with open('data/cub/all_sentences.txt', 'wt') as output_file:
    for sent in sentences:
        output_file.write(sent + '\n')

with open('data/cub/all_sentences_keys.txt', 'wt') as output_file:
    for k in sentences_keys:
        output_file.write(str(k) + '\n')
#
# phrases_list = []
# for sentence in tqdm(sentences):
#     doc = nlp(sentence)
#     sent = list(doc.sents)[0]
#     queue = [sent]
#     phrases = []
#     while len(queue) > 0:
#         segment = queue.pop(0)
#         for c in segment._.children:
#             phrases.append(str(c))
#             queue.append(c)
#     phrases_list.append(list(set(phrases)))
#
# with open('data/cub/phrases.pkl', 'wb') as output_file:
#     pickle.dump(phrases_list, output_file)

with open('data/cub/phrases.pkl', 'rb') as output_file:
    phrases_list = pickle.load(output_file)

with open('data/cub/kmeans.pkl', 'rb') as input_file:
    kmeans = pickle.load(input_file)

phrases_list_by_k = [[]] * max_k
phrases_set_by_k = [[]] * max_k
for i, l in enumerate(kmeans['labels'][max_k - 2]):
    phrases_list_by_k[l].extend(phrases_list[i])

for k_ in range(max_k):
    phrases_set_by_k[k_] = list(set(phrases_list_by_k[k_]))

phrases_roberta_by_k = [[]] * max_k
for k_ in tqdm(range(max_k)):
    print("k_", k_)
    for p in tqdm(phrases_set_by_k[k_]):
        phrases_roberta_by_k[k_].append(roberta.extract_features(roberta.encode(p)).detach().numpy())

data = {'phrases_set_by_k': phrases_set_by_k,
        'phrases_roberta_by_k': phrases_roberta_by_k}

with open('data/cub/phrases_roberta.base.pkl', 'wb') as output_file:
    pickle.dump(data, output_file)