import time
import numpy as np
import tensorflow as tf

class Density:
    def __init__(self, memory_bank):
        self.memory_bank = memory_bank

    def recompute_clusters(self):
        end = time.time()
        data = self.memory_bank.as_tensor()

        # Define the center (mean)
        center = np.mean(data, axis=0)

        # Compute Euclidean distances from the center
        distances = np.linalg.norm(data - center, axis=1)

        # Define a threshold (e.g., 90th percentile)
        threshold = np.percentile(distances, 90)

        # Assign cluster labels based on the threshold
        # Label 1: Close to center, Label 0: Far from center
        new_cluster_labels = (distances <= threshold).astype(int)
        new_cluster_labels = tf.expand_dims(new_cluster_labels, axis=0)

        print('Computing clusters time: {:.0f} ms'.format(1000* (time.time() - end)))
        
        return new_cluster_labels

