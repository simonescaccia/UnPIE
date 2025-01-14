import os
import time
import sklearn
import numpy as np
import tensorflow as tf

from model.cluster_density import Density
from model.cluster_km import Kmeans
from model.instance_model import InstanceModel
from model.memory_bank import MemoryBank
from model.self_loss import get_selfloss
from model.unpie_network import UnPIENetwork
from utils.print_utils import print_gpu_memory, print_separator, write_dict
from utils.vie_utils import tuple_get_one

import sys
np.set_printoptions(threshold=sys.maxsize)

class UnPIE():
    def __init__(self, params, args):

        self.params = params
        self.args = args

        self.cache_dir = self.params['save_params']['cache_dir'] # Set cache directory
        self.plot_save_path = os.path.join(self.cache_dir, self.params['save_params']['plot_dir'])
        os.system('mkdir -p %s' % self.cache_dir)
        os.system('mkdir -p %s' % self.plot_save_path)

        # Log files
        self.log_file_path = os.path.join(self.cache_dir, self.params['save_params']['train_log_file'])
        self.val_log_file_path = os.path.join(self.cache_dir, self.params['save_params']['val_log_file'])
        self.best_check_dir = os.path.join(self.cache_dir, 'best')
        self.last_check_dir = os.path.join(self.cache_dir, 'last')
        self.load_from_curr_exp = tf.train.latest_checkpoint(self.last_check_dir)
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
            **self.params['model_params']['model_func_params']
        )

        # Memory bank and clustering
        self.data_len = self.params['model_params']['model_func_params']['data_len']
        self.emb_dim = self.params['model_params']['model_func_params']['emb_dim']
        self.task = self.params['model_params']['model_func_params']['task']
        self.kmeans_k = self.params['model_params']['model_func_params']['kmeans_k']
        self.cluster_alg = self.params['model_params']['model_func_params']['cluster_alg']
        self.percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        
        num_kmeans_k = len(self.kmeans_k)
        num_clusters = len(self.percentiles) if self.cluster_alg == 'density' else num_kmeans_k

        self.memory_bank = MemoryBank(self.data_len, self.emb_dim)
        self.all_labels = self.params['datasets']['train']['dataloader'].dataset.y

        # Initialize lbl_init_values with 0 and 1s randomly
        lbl_init_values = np.random.randint(2, size=self.data_len)
        # Expand and tile lbl_init_values
        lbl_init_values = tf.tile(
            tf.expand_dims(lbl_init_values, axis=0),
            [num_clusters, 1]
        )
        
        # Cluster labels variable
        self.cluster_labels = tf.Variable(
            initial_value=lbl_init_values,
            trainable=False,
            dtype=tf.int64,
            name='cluster_labels'
        )

        # Optimizer
        opt_params = self.params['optimizer_params']
        self.learning_rate = self._get_learning_rate()
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate, 
            **opt_params)

        # Best accuracy
        self.best_val_acc = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # Checkpoint
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            cluster_labels=self.cluster_labels,
            memory_bank=self.memory_bank._bank,
            optimizer=self.optimizer,
            best_val_acc=self.best_val_acc,
            epoch=tf.Variable(0))
        self.last_check_manager = tf.train.CheckpointManager(self.checkpoint, self.last_check_dir, max_to_keep=1)
        self.best_check_manager = tf.train.CheckpointManager(self.checkpoint, self.best_check_dir, max_to_keep=1)

        self.monitor_metric = 'Accuracy s.l.' if self.task == 'SUP' else 'Accuracy u.f.l.'

        # Loss
        if self.task == 'SUP':
            self.sup_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def get_memory_bank(self):
        return self.memory_bank
    
    def get_cluster_labels(self):
        return self.cluster_labels

    def _build_network(self, x, b, c, a, y, i, train):
        model_params = self.params['model_params']
        model_func_params = model_params['model_func_params']
        res = self._build_output(x, b, c, a, y, i, train, **model_func_params)
        return res
    
    def _build_output(
        self,
        x, b, c, a, y, i, train,
        **kwargs):

        inputs = (x, b, c, a)

        if kwargs['task'] == 'SUP':
            output = self.model(inputs, train)

            if not train:
                return output
            
            loss = self.sup_loss_fn(y, output)
            return {'loss': loss}

        output = self.model(inputs, train)
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
        self.checkpoint.restore(ckpt_path).expect_partial()

    def _restore_model(self, checkpoint):
        if checkpoint:
            # Load current task training
            self._load_from_ckpt(checkpoint)
        elif self.params['load_params'] is not None:
            # Load previous task training
            load_dir_best = self._get_load_dir_best()
            ckpt_path = tf.train.latest_checkpoint(load_dir_best)
            self._load_from_ckpt(ckpt_path)
            self.checkpoint.epoch.assign(self.params['load_params']['step'])

    def _get_load_dir_best(self):
        split_cache_path = self.cache_dir.split(os.sep)
        split_cache_path[-1] = self.params['load_params']['task']
        load_dir = os.sep.join(split_cache_path)
        load_dir_best = os.path.join(load_dir, 'best')
        return load_dir_best

    def _get_learning_rate(self):
        lrs = self.params['learning_rate_params']['learning_rates']
        steps_per_epoch = self.params['learning_rate_params']['steps_per_epoch']
        boundaries = self.params['learning_rate_params']['boundaries']

        print('\nBoundaries: ', boundaries)
        print('Learning rates: ', lrs, '\n')

        if not boundaries:
            return lambda *_: lrs[0]
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

    @tf.function
    def _train_step(self, x, b, c, a, y, i):
        with tf.GradientTape() as tape:
            y_ = self._build_network(x, b, c, a, y, i, train=True)
            loss = self._build_loss(y_)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def _run_train_loop(self):
        clstr_update_per_epoch = self.params['train_params']['clstr_update_per_epoch']
        fre_plot_clusters = self.params['train_params']['fre_plot_clusters']
        fre_save_model = self.params['save_params']['fre_save_model']
        save_valid_freq = self.params['save_params']['save_valid_freq']
        train_dataloader = self.params['datasets']['train']['dataloader']
        num_steps = self.params['train_params']['num_steps']
        train_step = num_steps * self.checkpoint.epoch
        first_step = True

        # print("Saving clusters...")
        # self._save_memory_bank(self.memory_bank, train_dataloader.dataset.y, self.plot_save_path, self.checkpoint.epoch.numpy(), is_best=True)

        for epoch in range(self.checkpoint.epoch+1, self.params['train_params']['num_epochs']+1):

            for x, b, c, a, y, i in iter(train_dataloader):

                if clstr_update_per_epoch and first_step:
                    self._recompute_clusters()
                    first_step = False 

                x = tf.cast(x, tf.float32)
                b = tf.cast(b, tf.float32)
                c = tf.cast(c, tf.float32)
                a = tf.cast(a, tf.float32)
                y = tf.cast(y, tf.int32)
                i = tf.cast(i, tf.int32)

                train_step += 1
                self.start_time = time.time()
                
                # Optimize the model
                loss_value = self._train_step(x, b, c, a, y, i)

                duration = time.time() - self.start_time

                message = "{{'Epoch': {}, 'Step': {}, 'Time (ms)': {:.0f}, 'Loss': {}, 'Learning rate': {}}}".format(
                    epoch, train_step, 1000 * duration, loss_value, self.learning_rate(train_step))
                print(message)

                self.log_writer.write(message + '\n')

                # Recompute clusters
                if clstr_update_per_epoch and (train_step % (num_steps // clstr_update_per_epoch) == 0):
                   self._recompute_clusters()

            self.checkpoint.epoch.assign_add(1)

            # Compute validation
            if epoch % save_valid_freq == 0:
                val_result = self._run_inference_loop('val')
                val_result['Epoch'] = epoch
                self.val_log_writer.write(str(val_result) + '\n')
                print(val_result)

            # Save checkpoint
            if epoch % fre_save_model == 0:
                # if epoch % fre_plot_clusters == 0:
                #     self._save_memory_bank(self.memory_bank, train_dataloader.dataset.y, self.plot_save_path, epoch)
                if val_result[self.monitor_metric] > self.best_val_acc:
                    print('Saving model...')
                    self.best_val_acc.assign(val_result[self.monitor_metric])
                    self.best_check_manager.save(checkpoint_number=epoch)

                    # print("Saving clusters...")
                    # self._save_memory_bank(self.memory_bank, train_dataloader.dataset.y, self.plot_save_path, epoch, is_best=True)
                self.last_check_manager.save(checkpoint_number=epoch)

        # Check if a best model was saved for this task, else take the best model from the previous task
        if not os.path.exists(self.best_check_dir):
            load_dir_best = self._get_load_dir_best()
            # copy the best model from the previous task
            os.system('cp -r %s %s' % (load_dir_best, self.best_check_dir))

    def _recompute_clusters(self):
        print("Recomputing clusters...")
        if self.cluster_alg == 'kmeans':
            cluster_alg = Kmeans(self.kmeans_k, self.memory_bank)
        else:
            cluster_alg = Density(self.memory_bank, self.percentiles)
        self.cluster_labels.assign(cluster_alg.recompute_clusters())     
    
    def _save_memory_bank(self, memory_bank, y, save_path, epoch, is_best=False):
        memory_bank_dir = os.path.join(save_path, 'memory_bank')
        os.system('mkdir -p %s' % memory_bank_dir)

        # Save the true labels if they are not saved yet
        file_name = os.path.join(save_path, 'true_labels.npy')
        if not os.path.exists(file_name):
            np.save(file_name, y)

        # Remove previous best memory bank
        if is_best:
            postfix = 'best.npy'
             # Iterate through all items in the directory
            for file_name in os.listdir(memory_bank_dir):
                # Check if the item is a file and has the specified postfix
                if file_name.endswith(postfix) and os.path.isfile(os.path.join(memory_bank_dir, file_name)):
                    file_path = os.path.join(memory_bank_dir, file_name)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")

        # Save the memory bank
        best = '_best' if is_best else ''
        file_name = os.path.join(memory_bank_dir, f'epoch_{epoch}_memory_bank{best}.npy')
        np.save(file_name, memory_bank.as_tensor().numpy())


    def train(self):
        # Write args
        write_dict(self.args, os.path.join(self.cache_dir, 'args.txt')) 

        print_separator('Starting UnPIE training')
        self._restore_model(self.last_check_manager.latest_checkpoint)
        self._run_train_loop()
        self.log_writer.close()
        self.val_log_writer.close()
        print_separator('UnPIE training ended')


    # Validation functions
    def _build_inference_targets(self, y, i, outputs):
        target_params = self.params['inference_params']['targets']
        if self.task == 'SUP':
            targets = self._perf_func_sup(y, outputs)
        else:
            targets = self._perf_func_kNN(y, outputs, **target_params)
            # targets.update(self._perf_func_unsup(y, i, outputs))
        return targets

    def _inference_func(self, x, b, c, a, y, i):        
        outputs = self._build_network(x, b, c, a, y, i, train=False)
        targets = self._build_inference_targets(y, i, outputs)
        return targets

    def _run_inference_loop(self, dataloader_type):
        agg_res = None

        for x, b, c, a, y, i in iter(self.params['datasets'][dataloader_type]['dataloader']):

            x = tf.cast(x, tf.float32)
            b = tf.cast(b, tf.float32)
            c = tf.cast(c, tf.float32)
            a = tf.cast(a, tf.float32)
            y = tf.cast(y, tf.int32)
            i = tf.cast(i, tf.int32)

            res = self._inference_func(x, b, c, a, y, i)
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
    def _run_test_loop(self, test_type):
        test_result = self._run_inference_loop('test')
        self.test_log_writer.write(
                '%s: %s\n' % (f'test results {test_type} {self.checkpoint.epoch.numpy()}: ', str(test_result)))
        print(test_result)


    def test(self):
        print_separator('Starting UnPIE testing')
        self._restore_model(self.last_check_manager.latest_checkpoint)
        self._run_test_loop('last')
        self._restore_model(self.best_check_manager.latest_checkpoint)
        self._run_test_loop('best')
        self.test_log_writer.close()

        # tf.keras.utils.plot_model(self.model.gcn.build_graph(), to_file=os.path.join(self.cache_dir, 'scene_stgcn_model.png'), show_shapes=True, expand_nested=True, show_layer_names=False)
        # tf.keras.utils.plot_model(self.model.gcn.STGCN_layers_x[0].build_graph(), to_file=os.path.join(self.cache_dir, 'st_gcn_model.png'), show_shapes=True, expand_nested=True, show_layer_names=False)
        # tf.keras.utils.plot_model(self.model.gcn.STGCN_layers_x[0].sgcn.build_graph(), to_file=os.path.join(self.cache_dir, 'sgcn_model.png'), show_shapes=True, expand_nested=True, show_layer_names=False)

        print_separator('UnPIE testing ended')


    def _exclude_batch_norm(self, name):
        return 'batch_normalization' not in name

    # Loss functions
    def _reg_loss(self, loss, weight_decay):
        # Add weight decay to the loss.
        l2_loss = weight_decay * tf.add_n(
                [tf.nn.l2_loss(tf.cast(v, tf.float32))
                    for v in self.model.trainable_variables
                    if self._exclude_batch_norm(v.name)])
        loss_all = tf.add(loss, l2_loss)
        return loss_all
    

    # Compute metrics for unsupervised feature learning
    def _perf_func_kNN(
            self,
            y, output, 
            instance_t,
            k,
            num_classes):
        all_labels = self.all_labels
        curr_dist, _ = output
        all_labels = tuple_get_one(all_labels)
        top_dist, top_indices = tf.nn.top_k(curr_dist, k=k) # top k closest neighbor (highest similarity) for each test sample, top_dist: similarity score, top_indices: index of the closest neighbor
        top_labels = tf.gather(all_labels, top_indices) # retrieve the labels of the nearest neighbors
        top_labels_one_hot = tf.one_hot(top_labels, num_classes)
        top_prob = tf.exp(top_dist / instance_t)
        top_labels_one_hot *= tf.expand_dims(top_prob, axis=-1) # Each one-hot encoded label is weighted by its corresponding probability.
        top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1) # Averages the weighted probabilities across the kk neighbors for each test sample.
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
    
    def _perf_func_sup(
            self, 
            y, output):
        # output shape after sigmoid activation: (batch_size, 1)

        curr_output = tf.cast(output > 0.5, tf.int32)

        accuracy = sklearn.metrics.accuracy_score(y, curr_output)
        f1_score = sklearn.metrics.f1_score(y, curr_output, zero_division=0)
        # auc = sklearn.metrics.roc_auc_score(y, curr_pred)
        precision = sklearn.metrics.precision_score(y, curr_output, zero_division=0)
        return {
            'Accuracy s.l.': accuracy,
            'F1 s.l.': f1_score,
            # 'AUC u.f.l.': auc,
            'Precision s.l.': precision
        } 
    
    # Compute metrics for unsupervised learning
    def _perf_func_unsup(
            self,
            y, i, output):
        _, cluster_labels = output # cluster_labels: [num_kmeans_k, data_len] or [num_percentiles, data_len]        

        cluster_labels = tf.transpose(cluster_labels)
        cluster_labels = tf.gather(cluster_labels, i, axis=0) # [bs, num_kmeans_k] or [bs, num_percentiles]

        if self.cluster_alg == 'kmeans':
            # for each sample in bs get the most frequent cluster label
            y_pred = tf.argmax(
                tf.math.bincount(cluster_labels, axis=-1), # [bs, 2]
                axis=1
            ) # [bs]
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
        else: # Density
            metrics = {}
            # compute the metrics for each percentile
            for idx, percentile in enumerate(self.percentiles):
                y_pred = cluster_labels[:, idx]
                accuracy = sklearn.metrics.accuracy_score(y, y_pred)
                metrics[f'Accuracy u.l. {percentile}'] = accuracy
            return metrics
                
