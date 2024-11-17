import time
import faiss
import numpy as np
import tensorflow as tf

DEFAULT_SEED = 1234


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        (list: ids of data in each cluster, float: loss value)
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)

    # losses = faiss.vector_to_array(clus.obj)

    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])


    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


class Kmeans:
    def __init__(self, k, cluster_labels_callback, memory_bank_callback):
        self.k = k
        self.cluster_labels_callback = cluster_labels_callback
        self.memory_bank_callback = memory_bank_callback

    def recompute_clusters(self, verbose=True):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        data = self.memory_bank_callback().as_tensor()

        all_lables = []
        for k_idx, each_k in enumerate(self.k):
            # cluster the data
            I, _ = run_kmeans(data, each_k, 
                              verbose)
            new_clust_labels = np.asarray(I)
            all_lables.append(new_clust_labels)
        new_clust_labels = np.stack(all_lables, axis=0)

        if verbose:
            print('k-means time: {:.0f} ms'.format(1000* (time.time() - end)))
        
        self.cluster_labels_callback().assign(new_clust_labels)

