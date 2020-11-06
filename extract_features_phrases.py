import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance

import spacy
import nltk
import benepar
from benepar.spacy_plugin import BeneparComponent


benepar.download('benepar_en')
nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent('benepar_en'))

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

for sentence in sentences:
    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    queue = [sent]
    phrases = []
    while len(queue) > 0:
        segment = queue.pop(0)
        for c in sent._.children:
            phrases.append(str(c))
            queue.append(c)
    print(phrases)
    break