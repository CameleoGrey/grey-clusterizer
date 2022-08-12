
import numpy as np

from sklearn.cluster import KMeans

from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

x_raw, y = make_classification(n_samples=40000,
                                n_features=100,
                                n_informative=20,
                                n_redundant=10,
                                n_repeated=10,
                                n_classes=10,
                                n_clusters_per_class=1,
                                weights=None,
                                flip_y=0.05,
                                class_sep=3.0,
                                hypercube=True,
                                shift=0.5,
                                scale=1.0,
                                shuffle=True,
                                random_state=45)

x_raw = x_raw.astype(np.float32)

clusteriser = KMeans( n_clusters=10, init="k-means++", n_init=10, max_iter=300,
                      tol=1e-4, verbose=1, random_state=45, copy_x=True, algorithm="auto" )
clusteriser.fit( x_raw )

np.random.seed(45)
rand_ids = np.random.randint(0, x_raw.shape[0]-1, 10000)
x_subset = x_raw[rand_ids]
x_cluster_labels = clusteriser.predict(x_subset)

labels = x_cluster_labels
x_tsne = TSNE(n_components=2, n_iter=1000, perplexity=30.0, n_jobs=8, verbose=1, random_state=45).fit_transform( x_subset )
y_uniq = np.unique( labels )
for y_u in y_uniq:
    plt.scatter( x_tsne[ labels == y_u , 0], x_tsne[ labels == y_u , 1], s=2 )
plt.show()

print("done")

