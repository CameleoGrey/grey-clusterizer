
import numpy as np

from src.grey_clusterizer.HierarchicalGreyClusterizer import HierarchicalGreyClusterizer

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

clusteriser = HierarchicalGreyClusterizer(input_dim=x_raw.shape[1],
                                          latent_size=[256, 256], cluster_num=[10, 2], hidden_layer_dim=[512, 512],
                                          instance_feature_dim=[256, 256], hidden_layers_num=[3, 3], dropout_rate=[0.05, 0.05],
                                          device="cpu", save_dir=None, model_name=None)

clusteriser.fit( x_raw.copy(), batch_size=[256, 32], epochs=[100, 200], learning_rate=[0.0001, 0.0001],
            warm_up_epochs=[2, 2], early_stopping_rounds=[10, 10], backup_freq=[1000, 1000],
            instance_temperature=[0.5, 0.5], cluster_temperature=[1.0, 1.0],
            augmentation_power=[0.2, 0.2], verbose_freq=[2, 2] )

np.random.seed(45)
rand_ids = np.random.randint(0, x_raw.shape[0]-1, 10000)
x_subset = x_raw[rand_ids].copy()
x_cluster_labels = clusteriser.predict_batch_cluster_labels(x_subset.copy(), batch_size=256, encode_labels=True)

labels = x_cluster_labels
x_tsne = TSNE(n_components=2, n_iter=1000, perplexity=30.0, n_jobs=8, verbose=1, random_state=45).fit_transform( x_subset )
y_uniq = np.unique( labels )
for y_u in y_uniq:
    plt.scatter( x_tsne[ labels == y_u , 0], x_tsne[ labels == y_u , 1], s=2 )
    #for x, y in zip( x_tsne[ labels == y_u , 0], x_tsne[ labels == y_u , 1] ):
    #    plt.text( x, y, s=str(y_u) )
plt.show()

print("done")

