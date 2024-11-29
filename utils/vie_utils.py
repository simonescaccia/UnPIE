from __future__ import division, print_function, absolute_import
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


def online_keep_all(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(v)
    return agg_res


def tuple_get_one(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return x[0]
    return x

def plot_cluster(memory_bank, y, save_path, epoch):
    tsne = TSNE(n_components=2)
    data_2d = tsne.fit_transform(memory_bank.as_tensor().numpy())
    plot_cluster(data_2d, y, save_path, epoch, 'tsne')

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(memory_bank.as_tensor().numpy())
    plot_cluster(data_2d, y, save_path, epoch, 'pca')


def plot_cluster(data_2d, y, save_path, epoch, algorithm):
    # Create a figure for the true labels plot
    plt.figure(figsize=(7, 7))
    unique_labels = np.unique(y)
    for label in unique_labels:
        label_points = data_2d[y == label]
        plt.scatter(label_points[:, 0], label_points[:, 1], label=f'Label {label}')
    plt.title('True Labels Visualized in 2D')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()

    # Save the figure
    file_name = os.path.join(save_path, f'epoch_{epoch}_{algorithm}.png')
    plt.savefig(file_name)
    plt.close()  # Close the figure to avoid interactive display
