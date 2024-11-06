import os
import time
import tqdm
import tensorflow as tf

from model.instance_model import InstanceModel
from model.memory_bank import MemoryBank
from model.self_loss import get_selfloss
from model.unpie_network import UnPIENetwork
from utils.print_utils import print_separator
from utils.vie_utils import tuple_get_one

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

        self.nn_clusterings = []

        self.checkpoint = tf.train.Checkpoint(
            model=self.model, 
            step=tf.Variable(0))

        self.check_manager = tf.train.CheckpointManager(self.checkpoint, self.cache_dir, max_to_keep=5)

        self._init_and_restore()

    def _build_network(self, inputs, train):
        model_params = self.params['model_params']
        model_func_params = model_params['model_func_params']
        res = self._build_output(inputs, train, **model_func_params)
        if not train:
            return res
        outputs, clustering = res
        self.nn_clusterings.append(clustering)
        return outputs
    
    def _build_output(
        self,
        inputs, train,
        kmeans_k,
        task,
        **kwargs):

        inputs['a'] = tf.cast(inputs['a'], tf.float32)
        inputs['i'] = tf.cast(inputs['i'], tf.int32)

        data_len = kwargs.get('data_len')
        all_labels = tf.Variable(
            initial_value=tf.zeros(shape=(data_len,), dtype=tf.int64),
            trainable=False,
            dtype=tf.int64,
            name='all_labels'
        )

        memory_bank = MemoryBank(data_len, kwargs.get('emb_dim'))
        if task == 'LA':
            lbl_init_values = tf.range(data_len, dtype=tf.int64)
            no_kmeans_k = len(kmeans_k)
            lbl_init_values = tf.tile(
                    tf.expand_dims(lbl_init_values, axis=0),
                    [no_kmeans_k, 1])
            cluster_labels = tf.Variable(
                initial_value=lbl_init_values,
                trainable=False,
                dtype=tf.int64,
                name='cluster_labels'
            )

        output = self.model(
            inputs['x'], 
            inputs['a']
        )
        output = tf.nn.l2_normalize(output, axis=1)
        if not train:
            all_dist = memory_bank.get_all_dot_products(output)
            return [all_dist, all_labels]
        model_class = InstanceModel(
            inputs=inputs, output=output,
            memory_bank=memory_bank,
            **kwargs)
        nn_clustering = None
        other_losses = {}
        if task == 'LA':
            from .cluster_km import Kmeans
            nn_clustering = Kmeans(kmeans_k, memory_bank, cluster_labels)
            loss, _ = model_class.get_cluster_classification_loss(
                    cluster_labels)
        else:
            selfloss = get_selfloss(memory_bank, **kwargs)
            data_prob = model_class.compute_data_prob(selfloss)
            noise_prob = model_class.compute_noise_prob()
            losses = model_class.get_losses(data_prob, noise_prob)
            loss, loss_model, loss_noise = losses
            other_losses['loss_model'] = loss_model
            other_losses['loss_noise'] = loss_noise

        new_data_memory = model_class.updated_new_data_memory()
        ret_dict = {
            "loss": loss,
            "data_indx": inputs['i'],
            "memory_bank": memory_bank,
            "new_data_memory": new_data_memory,
            "all_labels": all_labels,
        }
        ret_dict.update(other_losses)
        return ret_dict, nn_clustering

    def _load_from_ckpt(self, ckpt_path):
        print('\nRestore from %s' % ckpt_path)
        self.checkpoint.restore(ckpt_path)

    def _init_and_restore(self):
        latest_checkpoint = self.check_manager.latest_checkpoint
        if latest_checkpoint:
            print("Here: latest_checkpoint", latest_checkpoint)
            # Load current task training
            self._load_from_ckpt(latest_checkpoint)
        elif self.params['load_params']['query'] is not None:
            print("Here: load_params query", self.params['load_params']['query'])
            # Load previous task training
            split_cache_path = self.cache_dir.split(os.sep)
            split_cache_path[-1] = self.params['load_params']['task']
            load_dir = os.sep.join(split_cache_path)
            ckpt_path = tf.train.latest_checkpoint(load_dir)
            print('\nRestore from %s' % ckpt_path)
            self._load_from_ckpt(ckpt_path)

    # Training functions
    def _build_train_opt(self, outputs):
        loss_params = self.params['loss_params']

        loss_retval = self._loss_func(
            outputs,
            **loss_params.get('loss_func_kwargs', {}))
        loss_retval = self._reg_loss(
            loss_retval,
            **loss_params.get('agg_func_kwargs', {}))
        
        lr_rate_params = self.params['learning_rate_params']
        learning_rate = self._get_lr_from_boundary_and_ramp_up(
            **lr_rate_params)

        opt_params = self.params['optimizer_params']
        train_opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, 
            **opt_params)

        return loss_retval, learning_rate, train_opt
            
    def _build_train_targets(self, outputs, loss_retval, learning_rate, train_opt):
        extra_targets_params = self.params['train_params']['targets']
        train_targets = self._rep_loss_func(
            outputs, 
            **extra_targets_params)

        train_targets['train_op'] = train_opt
        train_targets['loss'] = loss_retval
        train_targets['learning_rate'] = learning_rate

        return train_targets

    def _train_func(self, inputs):
        outputs = self._build_network(inputs, train=True)
        loss_retval, learning_rate, train_opt = self._build_train_opt(outputs) 
        train_targets = self._build_train_targets(outputs, loss_retval, learning_rate, train_opt)
        return train_targets

    def _run_train_loop(self):
        train_loop = self.params['train_params'].get('train_loop')
        train_func = self._train_func

        for _ in range(self.checkpoint.step+1, int(self.params['train_params']['num_steps']+1)):
            self.start_time = time.time()

            train_res = train_loop['func'](train_func, self.nn_clusterings, self.checkpoint.step)

            duration = time.time() - self.start_time

            self.checkpoint.step.assign_add(1)

            message = 'Step {} ({:.0f} ms) -- '\
                    .format(self.checkpoint.step.numpy(), 1000 * duration)
            rep_msg = ['{}: {:.4f}'.format(k, v) \
                    for k, v in train_res.items()
                    if k != 'train_op']
            message += ', '.join(rep_msg)
            print(message)

            if self.checkpoint.step % self.params['save_params']['fre_save_model'] == 0 \
                    and self.checkpoint.step > 0:
                print('Saving model...')
                self.check_manager.save(checkpoint_number=self.checkpoint.step)

            self.log_writer.write(message + '\n')
            if self.checkpoint.step % self.params['save_params']['save_valid_freq'] == 0:
                val_result = self._run_inference('validation_params')
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

    def _inference_func(self, inputs):
        inference_num_clips = self.params['inference_params']['targets']['inference_num_clips']
        x_shape = inputs['x'].shape
        inputs['x'] = tf.reshape(inputs['x'], 
            [x_shape[0]*inference_num_clips, x_shape[1]//inference_num_clips, x_shape[2], x_shape[3]])
        
        outputs = self._build_network(inputs, train=False)
        targets = self._build_inference_targets(inputs, outputs)
        return targets

    def _run_inference(self, params_type):
        agg_res = None
        num_steps = self.params[params_type]['num_steps']

        for _step in tqdm.trange(num_steps):
            res = self.params[params_type]['inference_loop']['func'](
                    self._inference_func)
            online_func = self.params['inference_params']['online_agg_func']
            agg_res = online_func(agg_res, res, _step)

        agg_func = self.params['inference_params']['agg_func']
        val_result = agg_func(agg_res)
        
        return val_result


    # Testing functions
    def _run_test_loop(self):
        test_result = self._run_inference('test_params')
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
        # Add weight decay to the loss.
        l2_loss = weight_decay * tf.add_n(
                [tf.nn.l2_loss(tf.cast(v, tf.float32))
                    for v in self.model.trainable_variables])
        loss_all = tf.add(loss, l2_loss)
        return loss_all
    
    def _loss_func(self, output):
        return output['loss']

    def _get_lr_from_boundary_and_ramp_up(
            self, 
            boundaries, 
            init_lr, target_lr, ramp_up_epoch,
            num_batches_per_epoch):
        curr_epoch  = tf.math.divide(
                tf.cast(self.checkpoint.step, tf.float32), 
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
        return curr_lr

    def _rep_loss_func(
            self,
            output
            ):
        ret_dict = {'loss_pure': output['loss']}
        for key, value in output.items():
            if key.startswith('loss_'):
                ret_dict[key] = value
        return ret_dict
    

    # Clustering functions
    def _perf_func_kNN(
            self,
            inputs, output, 
            instance_t,
            k, inference_num_clips,
            num_classes):
        curr_dist, all_labels = output
        all_labels = tuple_get_one(all_labels)
        top_dist, top_indices = tf.nn.top_k(curr_dist, k=k)
        top_labels = tf.gather(all_labels, top_indices)
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
                    tf.equal(curr_pred, inputs['y']),
                    tf.float32))
        return {'top1_{k}NN'.format(k=k): accuracy}

