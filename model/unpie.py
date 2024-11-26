import os
import time
import sklearn
import numpy as np
import tensorflow as tf

from model.instance_model import InstanceModel
from model.memory_bank import MemoryBank
from model.self_loss import get_selfloss
from model.unpie_network import UnPIENetwork
from utils.print_utils import print_separator, write_dict
from utils.vie_utils import tuple_get_one

import sys
np.set_printoptions(threshold=sys.maxsize)

class UnPIE():
    def __init__(self, params, args):

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
        # Write args
        write_dict(args, os.path.join(self.cache_dir, 'args.txt'))      
        
        # Model
        self.model = UnPIENetwork(
            **self.params['model_params']['model_func_params']
        )

        # Memory bank and clustering
        self.data_len = self.params['model_params']['model_func_params']['data_len']
        self.emb_dim = self.params['model_params']['model_func_params']['emb_dim']
        self.task = self.params['model_params']['model_func_params']['task']
        self.kmeans_k = self.params['model_params']['model_func_params']['kmeans_k']

        self.memory_bank = MemoryBank(self.data_len, self.emb_dim)
        self.nn_clustering = None

        self.all_labels = tf.Variable(
            initial_value=tf.zeros(shape=(self.data_len,), dtype=tf.int32),
            trainable=False,
            dtype=tf.int32,
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
            memory_bank=self.memory_bank._bank,
            epoch=tf.Variable(0))

        self.check_manager = tf.train.CheckpointManager(self.checkpoint, self.cache_dir, max_to_keep=1)

        # Optimizer
        opt_params = self.params['optimizer_params']
        self.learning_rate = self._get_learning_rate()
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate, 
            **opt_params)

        self._init_and_restore()

    def get_memory_bank(self):
        return self.memory_bank
    
    def get_cluster_labels(self):
        return self.cluster_labels

    def _build_network(self, x, b, a, i, train):
        model_params = self.params['model_params']
        model_func_params = model_params['model_func_params']
        res = self._build_output(x, b, a, i, train, **model_func_params)
        return res
    
    def _build_output(
        self,
        x, b, a, i, train,
        **kwargs):

        output = self.model(x, b, a, train)
        output = tf.nn.l2_normalize(output, axis=1)

        if not train:
            all_dist = self.memory_bank.get_all_dot_products(output) # cosine similarity (via matrix multiplication): similarity of a test sample to every training sample.
            return all_dist, self.cluster_labels

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

        tf.compat.v1.scatter_update(self.memory_bank.as_tensor(), i, model_class.updated_new_data_memory())
        
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

    def _get_learning_rate(self):
        lrs = self.params['learning_rate_params']['learning_rates']
        steps_per_epoch = self.params['learning_rate_params']['steps_per_epoch']
        boundaries = self.params['learning_rate_params']['boundaries']

        if boundaries is None:
            return lambda *args: lrs[0]
        boundaries = [x * steps_per_epoch for x in boundaries]
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=lrs
        )        

    # Training functions
    def _build_loss(self, outputs):
        loss_params = self.params['loss_params']

        loss_retval = outputs['loss']
        loss_retval = self._reg_loss(
            loss_retval,
            **loss_params.get('agg_func_kwargs', {}))
        
        return loss_retval

    def _loss(self, x, b, a, i):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = self._build_network(x, b, a, i, train=True)
        loss = self._build_loss(y_)

        return loss

    @tf.function
    def _grad(self, x, b, a, i):
        with tf.GradientTape() as tape:
            loss_value = self._loss(x, b, a, i)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def _run_train_loop(self):
        clstr_update_per_epoch = self.params['train_params']['clstr_update_per_epoch']
        fre_save_model = self.params['save_params']['fre_save_model']
        save_valid_freq = self.params['save_params']['save_valid_freq']
        train_dataloader = self.params['datasets']['train']['dataloader']
        num_steps = self.params['train_params']['num_steps']
        train_step = num_steps * self.checkpoint.epoch

        for epoch in range(self.checkpoint.epoch+1, self.params['train_params']['num_epochs']+1):

            for x, b, a, y, i in iter(train_dataloader):

                x = tf.cast(x, tf.float32)
                b = tf.cast(b, tf.float32)
                a = tf.cast(a, tf.float32)
                y = tf.cast(y, tf.int32)
                i = tf.cast(i, tf.int32)

                if epoch == 1:                  
                    # Validation and test purpose: compute distance
                    tf.compat.v1.scatter_update(self.all_labels, i, y) # collect all validation labels: first validation matric is not accurate

                train_step += 1
                self.start_time = time.time()
                
                # Optimize the model
                loss_value, grads = self._grad(x, b, a, i)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                duration = time.time() - self.start_time

                message = "{{'Epoch': {}, 'Step': {}, 'Time (ms)': {:.0f}, 'Loss': {}, 'Learning rate': {}}}".format(
                    epoch, train_step, 1000 * duration, loss_value, self.learning_rate(train_step))
                print(message)

                self.log_writer.write(message + '\n')

                # Recompute clusters for LA task: TODO choose right frequency, epoch/batch/multi-batch
                if self.nn_clustering is not None and train_step % (num_steps // clstr_update_per_epoch) == 0:
                    print("Recomputing clusters...")
                    self.nn_clustering.recompute_clusters()

            self.checkpoint.epoch.assign_add(1)

            # Compute validation
            if epoch % save_valid_freq == 0:
                val_result = self._run_inference_loop('val')
                val_result['Epoch'] = epoch
                self.val_log_writer.write(str(val_result) + '\n')
                print(val_result)
                self.val_log_writer.close()
                self.val_log_writer = open(self.val_log_file_path, 'a+')

            # Save checkpoint
            if epoch % fre_save_model == 0:
                print('Saving model...')
                self.check_manager.save(checkpoint_number=epoch)


    def train(self):
        print_separator('Starting UnPIE training')
        self._run_train_loop()
        print_separator('UnPIE training ended')


    # Validation functions
    def _build_inference_targets(self, y, i, outputs):
        target_params = self.params['inference_params']['targets']
        targets = self._perf_func_kNN(y, outputs, **target_params)
        targets.update(self._perf_func_unsup(y, i, outputs))
        return targets

    def _inference_func(self, x, b, a, y, i):        
        outputs = self._build_network(x, b, a, i, train=False)
        targets = self._build_inference_targets(y, i, outputs)
        return targets

    def _run_inference_loop(self, dataloader_type):
        agg_res = None

        for x, b, a, y, i in iter(self.params['datasets'][dataloader_type]['dataloader']):

            x = tf.cast(x, tf.float32)
            b = tf.cast(b, tf.float32)
            a = tf.cast(a, tf.float32)
            y = tf.cast(y, tf.int32)
            i = tf.cast(i, tf.int32)

            res = self._inference_func(x, b, a, y, i)
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
    

    # Compute metrics for unsupervised feature learning
    def _perf_func_kNN(
            self,
            y, output, 
            instance_t,
            k, inference_num_clips,
            num_classes):
        all_labels = self.all_labels
        curr_dist, _ = output
        all_labels = tuple_get_one(all_labels)
        top_dist, top_indices = tf.nn.top_k(curr_dist, k=k) # top k closest neighbor (highest similarity) for each test sample, top_dist: similarity score, top_indices: index of the closest neighbor
        top_labels = tf.gather(all_labels, top_indices) # retrieve the labels of the nearest neighbors
        top_labels_one_hot = tf.one_hot(top_labels, num_classes)
        top_prob = tf.exp(top_dist / instance_t)
        top_labels_one_hot *= tf.expand_dims(top_prob, axis=-1)
        top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
        _, curr_pred = tf.nn.top_k(top_labels_one_hot, k=1)
        curr_pred = tf.squeeze(tf.cast(curr_pred, tf.int32), axis=1)
        accuracy = sklearn.metrics.accuracy_score(y, curr_pred)
        f1_score = sklearn.metrics.f1_score(y, curr_pred, zero_division=0)
        # auc = sklearn.metrics.roc_auc_score(y, curr_pred)
        precision = sklearn.metrics.precision_score(y, curr_pred, zero_division=0)
        return {
            'Accuracy u.f.l.': accuracy,
            'F1 u.f.l.': f1_score,
            # 'AUC u.f.l.': auc,
            'Precision u.f.l.': precision
        }
    
    
    # Compute metrics for unsupervised learning
    def _perf_func_unsup(
            slf,
            y, i, output):
        _, cluster_labels = output
        cluster_labels = tf.squeeze(cluster_labels, axis=0)
        y_pred = tf.gather(cluster_labels, i)
        accuracy = sklearn.metrics.accuracy_score(y, y_pred)
        f1_score = sklearn.metrics.f1_score(y, y_pred, zero_division=0)
        # auc = sklearn.metrics.roc_auc_score(y, y_pred)
        precision = sklearn.metrics.precision_score(y, y_pred, zero_division=0)
        return {
            'Accuracy u.l.': accuracy,
            'F1 u.l.': f1_score,
            # 'AUC u.l.': auc,
            'Precision u.l.': precision
        }