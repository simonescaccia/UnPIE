import os
import time
import tqdm
import numpy as np
import tensorflow as tf

from model.instance_model import InstanceModel
from model.memory_bank import MemoryBank
from model.self_loss import get_selfloss
from model.unpie_network import UnPIENetwork
from utils.print_utils import print_separator
from utils.vie_utils import tuple_get_one

import sys
np.set_printoptions(threshold=sys.maxsize)

class UnPIE():
    def __init__(self, params):

        self.params = params

        # Log files
        self.cache_dir = self.params['save_params']['cache_dir'] # Set cache directory
        self.log_file_path = os.path.join(self.cache_dir, self.params['save_params']['train_log_file'])
        self.val_log_file_path = os.path.join(self.cache_dir, self.params['save_params']['val_log_file'])
        os.system('mkdir -p %s' % self.cache_dir)
        self.load_from_curr_exp = tf.train.latest_checkpoint(self.cache_dir)
        if not self.load_from_curr_exp: # if no checkpoint is found then create a new log file
            self.log_writer = open(self.log_file_path, 'w')
            self.val_log_writer = open(self.val_log_file_path, 'w')
        else: # if checkpoint is found then append to the existing log file
            self.log_writer = open(self.log_file_path, 'a+')
            self.val_log_writer = open(self.val_log_file_path, 'a+')
        if self.params['is_test']:
            self.test_log_file_path = os.path.join(self.cache_dir, self.params['save_params']['test_log_file'])
            self.test_log_writer = open(self.test_log_file_path, 'w')
        
        # Model
        self.model = UnPIENetwork(
            self.params['model_params']['model_func_params']['middle_dim'],
            self.params['model_params']['model_func_params']['emb_dim'],
        )

        # Memory bank and clustering
        self.data_len = self.params['model_params']['model_func_params']['data_len']
        self.emb_dim = self.params['model_params']['model_func_params']['emb_dim']
        self.task = self.params['model_params']['model_func_params']['task']
        self.kmeans_k = self.params['model_params']['model_func_params']['kmeans_k']
        self.memory_bank = MemoryBank(self.data_len, self.emb_dim)
        self.nn_clustering = None

        self.all_labels = tf.Variable(
            initial_value=tf.zeros(shape=(self.data_len,), dtype=tf.int64),
            trainable=False,
            dtype=tf.int64,
            name='all_labels'
        )
        # Initialize lbl_init_values with a range
        lbl_init_values = tf.range(self.data_len, dtype=tf.int64)
        no_kmeans_k = len(self.kmeans_k)
        # Expand and tile lbl_init_values
        lbl_init_values = tf.tile(
            tf.expand_dims(lbl_init_values, axis=0),
            [no_kmeans_k, 1]
        )
        
        # Cluster labels variable
        self.cluster_labels = tf.Variable(
            initial_value=lbl_init_values,
            trainable=False,
            dtype=tf.int64,
            name='cluster_labels'
        )

        if self.task == 'LA':
            from .cluster_km import Kmeans
            self.nn_clustering = Kmeans(self.kmeans_k, self.get_cluster_labels, self.get_memory_bank)

        # Checkpoint
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            all_labels = self.all_labels,
            cluster_labels=self.cluster_labels,
            epoch=tf.Variable(0))

        self.check_manager = tf.train.CheckpointManager(self.checkpoint, self.cache_dir, max_to_keep=1)

        # Optimizer: TODO support rump up
        lr_rate_params = self.params['learning_rate_params']
        self.learning_rate = self._get_lr_from_boundary_and_ramp_up(
            **lr_rate_params)

        opt_params = self.params['optimizer_params']
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate, 
            **opt_params)


        self._init_and_restore()

    def get_memory_bank(self):
        return self.memory_bank
    
    def get_cluster_labels(self):
        return self.cluster_labels

    def _build_network(self, x, a, y, i, train):
        model_params = self.params['model_params']
        model_func_params = model_params['model_func_params']
        res = self._build_output(x, a, y, i, train, **model_func_params)
        return res
    
    def _build_output(
        self,
        x, a, y, i, train,
        **kwargs):

        a = tf.cast(a, tf.float32)
        i = tf.cast(i, tf.int32)

        output = self.model(
            x, 
            a
        )
        output = tf.nn.l2_normalize(output, axis=1)

        if not train:
            all_dist = self.memory_bank.get_all_dot_products(output) # cosine similarity (via matrix multiplication): similarity of a test sample to every training sample.
            return all_dist

        # Training: compute loss
        model_class = InstanceModel(
            i=i, output=output,
            memory_bank=self.memory_bank,
            **kwargs)
        other_losses = {}
        if self.task == 'LA':
            loss, _ = model_class.get_cluster_classification_loss(self.cluster_labels)
        else:
            selfloss = get_selfloss(self.memory_bank, **kwargs)
            data_prob = model_class.compute_data_prob(selfloss)
            noise_prob = model_class.compute_noise_prob()
            losses = model_class.get_losses(data_prob, noise_prob)
            loss, loss_model, loss_noise = losses
            other_losses['loss_model'] = loss_model
            other_losses['loss_noise'] = loss_noise

        tf.compat.v1.scatter_update(
                    self.memory_bank.as_tensor(), i, model_class.updated_new_data_memory())
        
        ret_dict = {
            "loss": loss,
        }
        ret_dict.update(other_losses)
        return ret_dict

    def _load_from_ckpt(self, ckpt_path):
        print('\nRestore from %s' % ckpt_path)
        self.checkpoint.restore(ckpt_path)

    def _init_and_restore(self):
        latest_checkpoint = self.check_manager.latest_checkpoint
        if latest_checkpoint:
            # Load current task training
            self._load_from_ckpt(latest_checkpoint)
        elif self.params['load_params'] is not None:
            # Load previous task training
            split_cache_path = self.cache_dir.split(os.sep)
            split_cache_path[-1] = self.params['load_params']['task']
            load_dir = os.sep.join(split_cache_path)
            ckpt_path = tf.train.latest_checkpoint(load_dir)
            self._load_from_ckpt(ckpt_path)

    # Training functions
    def _build_loss(self, outputs):
        loss_params = self.params['loss_params']

        loss_retval = outputs['loss']
        loss_retval = self._reg_loss(
            loss_retval,
            **loss_params.get('agg_func_kwargs', {}))
        
        return loss_retval

    def _loss(self, x, a, y, i):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = self._build_network(x, a, y, i, train=True)
        loss = self._build_loss(y_)

        return loss

    def _grad(self, x, a, y, i):
        with tf.GradientTape() as tape:
            loss_value = self._loss(x, a, y, i)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def _run_train_loop(self):

        for epoch in range(self.checkpoint.epoch+1, self.params['train_params']['num_epochs']+1):
            if epoch == 2:
                print("self.all_labels: ", self.all_labels)
            train_step = 0

            for x, a, y, i in iter(self.params['datasets']['train']['dataloader']):

                if epoch == 1:                  
                    # Validation and test purpose: compute distance
                    tf.compat.v1.scatter_update(self.all_labels, i, y) # collect all validation labels: first validation matric is not accurate

                train_step += 1
                self.start_time = time.time()
                
                # Optimize the model
                loss_value, grads = self._grad(x, a, y, i)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                duration = time.time() - self.start_time

                message = 'Epoch {}, Step {} ({:.0f} ms) -- Loss {}'.format(
                    epoch, train_step, 1000 * duration, loss_value)
                print(message)

                self.log_writer.write(message + '\n')

                # Recompute clusters for LA task: TODO choose right frequency, epoch/batch/multi-batch
                if self.nn_clustering is not None:
                    print("Recomputing clusters...")
                    self.nn_clusterings.recompute_clusters()

            self.checkpoint.epoch.assign_add(1)

            # Save checkpoint
            if epoch % self.params['save_params']['fre_save_model'] == 0:
                print('Saving model...')
                self.check_manager.save(checkpoint_number=epoch)
            
            # Compute validation
            if epoch % self.params['save_params']['save_valid_freq'] == 0:
                val_result = self._run_inference_loop('val')
                self.val_log_writer.write(
                        '%s: %s\n' % ('validation results: ', str(val_result)))
                print(val_result)
                self.val_log_writer.close()
                self.val_log_writer = open(self.val_log_file_path, 'a+')


    def train(self):
        print_separator('Starting UnPIE training')
        self._run_train_loop()
        print_separator('UnPIE training ended')


    # Validation functions
    def _build_inference_targets(self, inputs, outputs):
        target_params = self.params['inference_params']['targets']
        targets = self._perf_func_kNN(inputs, outputs, **target_params)
        return targets

    def _inference_func(self, x, a, y, i):
        inference_num_clips = self.params['inference_params']['targets']['inference_num_clips']
        x_shape = x.shape
        x = tf.reshape(x, 
            [x_shape[0]*inference_num_clips, x_shape[1]//inference_num_clips, x_shape[2], x_shape[3]])
        
        outputs = self._build_network(x, a, y, i, train=False)
        targets = self._build_inference_targets(y, outputs)
        return targets

    def _run_inference_loop(self, dataloader_type):
        agg_res = None

        for x, a, y, i in iter(self.params['datasets'][dataloader_type]['dataloader']):
            res = self._inference_func(x, a, y, i)
            agg_res = self._online_agg(agg_res, res)

        val_result = self._agg_func(agg_res)
        
        return val_result
    
    def _agg_func(self, x):
        return {k: np.mean(v) for k, v in x.items()}
    
    def _online_agg(self, agg_res, res):
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            agg_res[k].append(np.mean(v))
            # agg_res[k].append(v)
        return agg_res 


    # Testing functions
    def _run_test_loop(self):
        test_result = self._run_inference_loop('test')
        self.test_log_writer.write(
                '%s: %s\n' % ('test results: ', str(test_result)))
        print(test_result)
        self.test_log_writer.close()

    def test(self):
        print_separator('Starting UnPIE testing')
        self._run_test_loop()
        print_separator('UnPIE testing ended')


    # Loss functions
    def _reg_loss(self, loss, weight_decay):
        def exclude_batch_norm(name):
            return 'batch_normalization' not in name
        # Add weight decay to the loss.
        l2_loss = weight_decay * tf.add_n(
                [tf.nn.l2_loss(tf.cast(v, tf.float32))
                    for v in self.model.trainable_variables
                    if exclude_batch_norm(v.name)])
        loss_all = tf.add(loss, l2_loss)
        return loss_all

    def _get_lr_from_boundary_and_ramp_up(
            self, 
            boundaries, 
            init_lr, target_lr, ramp_up_epoch,
            num_batches_per_epoch):
        curr_epoch  = tf.math.divide(
                tf.cast(self.checkpoint.epoch, tf.float32), 
                tf.cast(num_batches_per_epoch, tf.float32))
        curr_phase = (tf.minimum(curr_epoch/float(ramp_up_epoch), 1))
        curr_lr = init_lr + (target_lr-init_lr) * curr_phase

        if boundaries is not None:
            boundaries = boundaries.split(',')
            boundaries = [int(each_boundary) for each_boundary in boundaries]

            all_lrs = [
                    curr_lr * (0.1 ** drop_level) \
                    for drop_level in range(len(boundaries) + 1)]

            curr_lr = tf.compat.v1.train.piecewise_constant(
                    x=self.checkpoint.step,
                    boundaries=boundaries, values=all_lrs)
        
            curr_lr = curr_lr() # eager execution compatibility

        return curr_lr

    # Inference function to compute metrics
    def _perf_func_kNN(
            self,
            y, output, 
            instance_t,
            k, inference_num_clips,
            num_classes):
        all_labels = self.all_labels
        curr_dist = output
        all_labels = tuple_get_one(all_labels)
        top_dist, top_indices = tf.nn.top_k(curr_dist, k=k) # top k closest neighbor (highest similarity) for each test sample, top_dist: similarity score, top_indices: index of the closest neighbor
        top_labels = tf.gather(all_labels, top_indices) # retrieve the labels of the nearest neighbors
        top_labels_one_hot = tf.one_hot(top_labels, num_classes)
        top_prob = tf.exp(top_dist / instance_t)
        top_labels_one_hot *= tf.expand_dims(top_prob, axis=-1)
        top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
        top_labels_one_hot = tf.reshape(
                top_labels_one_hot,
                [-1, inference_num_clips, num_classes])
        top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
        _, curr_pred = tf.nn.top_k(top_labels_one_hot, k=1)
        curr_pred = tf.squeeze(tf.cast(curr_pred, tf.int64), axis=1)
        print("curr_pred: ", curr_pred)
        accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(curr_pred, y),
                    tf.float32))
        return {'top1_{k}NN'.format(k=k): accuracy}
