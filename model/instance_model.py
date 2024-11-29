from __future__ import division, print_function, absolute_import
import tensorflow as tf
from .self_loss import assert_shape

def repeat_1d_tensor(t, num_reps):
    ret = tf.tile(tf.expand_dims(t, axis=1), (1, num_reps))
    return ret


class InstanceModel(object):
    def __init__(self,
                 i, output,
                 memory_bank,
                 instance_k,
                 instance_t=0.07,
                 instance_m=0.5,
                 **kwargs):
        self.i = i
        self.embed_output = output
        self.batch_size, self.out_dim = self.embed_output.get_shape().as_list()
        self.memory_bank = memory_bank

        self.instance_data_len = memory_bank.size
        self.instance_k = instance_k
        self.instance_t = instance_t
        self.instance_m = instance_m

    def _softmax(self, dot_prods):
        instance_Z = tf.constant(
            2876934.2 / 1281167 * self.instance_data_len,
            dtype=tf.float32)
        return tf.exp(dot_prods / self.instance_t) / instance_Z

    def updated_new_data_memory(self):
        data_indx = self.i # [bs]
        data_memory = self.memory_bank.at_idxs(data_indx)
        new_data_memory = (data_memory * self.instance_m
                           + (1 - self.instance_m) * self.embed_output)
        return tf.nn.l2_normalize(new_data_memory, axis=1)

    def _get_lbl_equal(self, each_k_idx, cluster_labels, top_idxs, k):
        batch_labels = tf.gather(
                cluster_labels[each_k_idx], 
                self.i)
        if k > 0:
            top_cluster_labels = tf.gather(cluster_labels[each_k_idx], top_idxs)
            batch_labels = repeat_1d_tensor(batch_labels, k)
            curr_equal = tf.equal(batch_labels, top_cluster_labels)
        else:
            curr_equal = tf.equal(
                    tf.expand_dims(batch_labels, axis=1), 
                    tf.expand_dims(cluster_labels[each_k_idx], axis=0))
        return curr_equal

    def _get_prob_from_equal(self, curr_equal, exponents):
        probs = tf.reduce_sum(
            tf.where(
                curr_equal,
                x=exponents, y=tf.zeros_like(exponents),
            ), axis=1)
        probs /= tf.reduce_sum(exponents, axis=1)
        return probs

    def get_cluster_classification_loss(
            self, cluster_labels):
        '''
        Aim: Compute the loss starting from the cluster labels
        '''
        k = self.instance_k
        # ignore all but the top k nearest examples
        all_dps = self.memory_bank.get_all_dot_products(self.embed_output)
        top_dps, top_idxs = tf.nn.top_k(all_dps, k=k, sorted=False)
        if k > 0:
            exponents = self._softmax(top_dps)
        else:
            exponents = self._softmax(all_dps)

        no_kmeans = cluster_labels.get_shape().as_list()[0]
        all_equal = None
        for each_k_idx in range(no_kmeans):
            curr_equal = self._get_lbl_equal(
                    each_k_idx, cluster_labels, top_idxs, k)

            if all_equal is None:
                all_equal = curr_equal
            else:
                all_equal = tf.logical_or(all_equal, curr_equal)
        probs = self._get_prob_from_equal(all_equal, exponents)

        assert_shape(probs, [self.batch_size])
        loss = -tf.reduce_mean(tf.math.log(probs + 1e-7))
        return loss, self.i

    def compute_data_prob(self, selfloss):
        data_indx = self.i
        logits = selfloss.get_closeness(data_indx, self.embed_output)
        return self._softmax(logits)

    def compute_noise_prob(self):
        noise_indx = tf.random.uniform(
            shape=(self.batch_size, self.instance_k),
            minval=0,
            maxval=self.instance_data_len,
            dtype=tf.int64)
        noise_probs = self._softmax(
            self.memory_bank.get_dot_products(self.embed_output, noise_indx))
        return noise_probs

    def get_losses(self, data_prob, noise_prob):
        assert_shape(data_prob, [self.batch_size])
        assert_shape(noise_prob, [self.batch_size, self.instance_k])

        base_prob = 1.0 / self.instance_data_len
        eps = 1e-7
        ## Pmt
        data_div = data_prob + (self.instance_k*base_prob + eps)
        ln_data = tf.math.log(data_prob / data_div)
        ## Pon
        noise_div = noise_prob + (self.instance_k*base_prob + eps)
        ln_noise = tf.math.log((self.instance_k*base_prob) / noise_div)

        curr_loss = -(tf.reduce_sum(ln_data) \
                      + tf.reduce_sum(ln_noise)) / self.batch_size
        return curr_loss, \
            -tf.reduce_sum(ln_data)/self.batch_size, \
            -tf.reduce_sum(ln_noise)/self.batch_size
