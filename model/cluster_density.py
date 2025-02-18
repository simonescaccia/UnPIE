import time
import numpy as np
import tensorflow as tf

class Density:
    def __init__(self, memory_bank, percentiles):
        self.memory_bank = memory_bank
        self.percentiles = percentiles

    def recompute_clusters(self):
        end = time.time()
        data = self.memory_bank.as_tensor() # [size, dim]

        # Define the center (mean) of the data
        center = tf.reduce_mean(data, axis=0) # [dim]
        center = tf.expand_dims(center, axis=0) # [1, dim]

        # Compute distances from the center using cosine similarity
        distances = tf.matmul(center, tf.transpose(data, [1, 0])) # [1, dim] * [dim, size] = [1, size]
        distances = tf.squeeze(distances, axis=0)

        all_cluster_labels = []
        for percentile in self.percentiles:    
            # Define a threshold (e.g., 90th percentile)
            threshold = np.percentile(distances, percentile)

            # Assign cluster labels based on the threshold
            # Label 1: Close to center, Label 0: Far from center
            new_cluster_labels = tf.cast(tf.greater(threshold, distances), tf.int32)

            all_cluster_labels.append(new_cluster_labels)

        print('Computing clusters time: {:.0f} ms'.format(1000* (time.time() - end)))
        
        return all_cluster_labels # [len_percentiles, size]

