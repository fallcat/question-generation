import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance


max_k = 25

data_features_file = pickle.load(open('data/cub/descriptions_roberta.base.pkl', 'rb'))
data_features = data_features_file['data_features']
flattened_features = data_features_file['flattened_features']
flattened_features_keys = data_features_file['flattened_features_keys']

with open('data/cub/classes_w_descriptions_aab_ebird.tsv', 'rt') as input_file:
    keys = [line.strip().split('\t')[0] for line in input_file.readlines()]

X = np.array(flattened_features)

pca = PCA(n_components=2)
v = pca.fit(np.transpose(X)).components_

kmeans_labels_list = []
kmeans_cluster_centers_list = []
n_iters_list = []

scores = [0] * (max_k-1)
for k in tqdm(range(2, max_k + 1)):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    kmeans_labels_list.append(kmeans.labels_)
    kmeans_cluster_centers_list.append(kmeans.cluster_centers_)
    n_iters_list.append(kmeans.n_iter_)
    plt.figure()
    plt.scatter(v[0], v[1], c=kmeans.labels_, s=20)
    plt.title("CUB Descriptions Clusters (k = " + str(k) + ")")
    plt.savefig(f'figs/cub/k{k}.pdf')
    plt.close()

    distortion = 0
    for i, x in enumerate(X):
        distortion += distance.euclidean(x, kmeans.cluster_centers_[kmeans.labels_[i]]) ** 2
    scores[k - 2] = distortion

plt.figure()
plt.plot(range(max_k-1), scores, label='distortion')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('K-Means Distortion with K ranging from 2-25')
plt.ylabel('Distortion')
plt.xlabel('K Value')
plt.xticks(range(max_k-1), range(2,max_k+1))
plt.savefig('figs/cub/distortion.png')
plt.close()

kmeans_result = {'labels': kmeans_labels_list,
                 'centroids': kmeans_cluster_centers_list,
                 'iters': n_iters_list}
with open('data/cub/kmeans.pkl', 'wb') as output_file:
    pickle.dump(kmeans_result, output_file)