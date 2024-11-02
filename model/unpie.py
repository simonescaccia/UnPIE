import os
import time
import tqdm
import numpy as np
import tensorflow as tf

from utils.print_utils import print_separator
from utils.vie_utils import tuple_get_one

class UnPIE(tf.Module):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        self.params = params

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

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)     

    def build_network(self, inputs, train):
        model_params = self.params['model_params']
        model_func_params = model_params['model_func_params']
        func = model_params.pop('func')
        outputs, _ = func(
                inputs=inputs,
                train=train,
                **model_func_params)
        return outputs

    def load_from_ckpt(self, ckpt_path):
        print('\nRestore from %s' % ckpt_path)
        self.saver.restore(self.sess, ckpt_path)

    def build_sess_and_saver(self):
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options,
                ))
        self.sess = sess
        self.saver = tf.compat.v1.train.Saver()

    def init_and_restore(self):
        init_op_global = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op_global)
        init_op_local = tf.compat.v1.local_variables_initializer()
        self.sess.run(init_op_local)

        if self.load_from_curr_exp:
            self.load_from_ckpt(self.load_from_curr_exp)
        else:
            split_cache_path = self.cache_dir.split(os.sep)
            split_cache_path[-1] = self.params['load_params']['exp_id']
            load_dir = os.sep.join(split_cache_path)
            if self.params['load_params']['query']:
                ckpt_path = os.path.join(
                        load_dir, 
                        'model.ckpt-%i' % self.params['load_params']['query']['step'])
            else:
                ckpt_path = tf.train.latest_checkpoint(load_dir)
            if ckpt_path:
                print('\nRestore from %s' % ckpt_path)
                #self.load_from_ckpt(ckpt_path)
                reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
                saved_var_shapes = reader.get_variable_to_shape_map()

                all_vars = tf.compat.v1.global_variables()
                all_var_list = {v.op.name: v for v in all_vars}
                filtered_var_list = {}
                for name, var in all_var_list.items():
                    if name in saved_var_shapes:
                        curr_shape = var.get_shape().as_list()
                        saved_shape = saved_var_shapes[name]
                        if (curr_shape == saved_shape):
                            filtered_var_list[name] = var
                        else:
                            print('Shape mismatch for %s: ' % name \
                                    + str(curr_shape) \
                                    + str(saved_shape))
                _load_saver = tf.compat.v1.train.Saver(var_list=filtered_var_list)
                _load_saver.restore(self.sess, ckpt_path)

    def build_test_inputs(self, test_key):
        data_params = self.params['test_params'][test_key]['data_params']
        func = data_params.pop('func')
        test_inputs = func(**data_params)
        return test_inputs
    
    def build_test_network(self, test_key, test_inputs):
        with tf.name_scope('test/' + test_key):
            test_outputs = self.build_network(test_inputs, False)
        return test_outputs

    def build_test(self):
        tf.compat.v1.get_variable_scope().reuse_variables()
        self.all_test_targets = {}
        for each_val_key in self.params['test_params']:
            test_inputs = self.build_test_inputs(each_val_key)
            test_outputs = self.build_test_network(each_val_key, test_inputs)
            test_targets = self.build_test_targets(
                    each_val_key, test_inputs, test_outputs)
            self.all_test_targets[each_val_key] = test_targets           

    def build_model(self):
        if self.params['is_test']:
            self.build_test()

        self.build_sess_and_saver()
        self.init_and_restore()

    # Training functions
    def build_train_opt(self, outputs):
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
        train_opt = tf.keras.optimizers.SDG(
            learning_rate=learning_rate, 
            **opt_params)
        
        return loss_retval, learning_rate, train_opt 
            
    def build_train_targets(self, outputs, loss_retval, learning_rate, train_opt):
        extra_targets_params = self.params['train_params']['targets']
        train_targets = self._rep_loss_func(
            outputs, 
            **extra_targets_params)

        train_targets['train_op'] = train_opt
        train_targets['loss'] = loss_retval
        train_targets['learning_rate'] = learning_rate

        return train_targets

    def train_func(self, inputs):
        outputs = self.build_network(inputs, train=True)
        loss_retval, learning_rate, train_opt = self.build_train_opt(outputs)     
        train_targets = self.build_train_targets(outputs, loss_retval, learning_rate, train_opt)
        return train_targets

    def _run_train_loop(self):
        start_step = self.global_step
        train_loop = self.params['train_params'].get('train_loop')
        train_func = self.train_func

        for curr_step in range(start_step, int(self.params['train_params']['num_steps']+1)):
            self.start_time = time.time()
            train_res = train_loop['func'](train_func)

            duration = time.time() - self.start_time

            message = 'Step {} ({:.0f} ms) -- '\
                    .format(curr_step, 1000 * duration)
            rep_msg = ['{}: {:.4f}'.format(k, v) \
                    for k, v in train_res.items()
                    if k != 'train_op']
            message += ', '.join(rep_msg)
            print(message)

            if curr_step % self.params['save_params']['fre_save_model'] == 0 \
                    and curr_step > 0:
                print('Saving model...')
                self.saver.save(
                        self.sess, 
                        os.path.join(
                            self.cache_dir,
                            'model.ckpt'), 
                        global_step=curr_step)

            self.log_writer.write(message + '\n')
            if curr_step % self.params['save_params']['save_metrics_freq'] == 0:
                self.log_writer.close()
                self.log_writer = open(self.log_file_path, 'a+')

            if curr_step % self.params['save_params']['save_valid_freq'] == 0:
                val_result = self.run_inference()
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
    def build_inference_targets(self, inputs, outputs):
        target_params = self.params['inference_params']['targets']
        targets = self._perf_func_kNN(inputs, outputs, **target_params)
        return targets

    def inference_func(self, inputs):
        outputs = self.build_network(inputs, train=False)
        targets = self.build_inference_targets(inputs, outputs)
        return targets

    def run_inference(self, num_steps):
        agg_res = None

        for _step in tqdm.trange(num_steps):
            res = self.params['inference_params']['inference_loop']['func'](
                    self.inference_func)
            online_func = self.params['inference_params']['online_agg_func']
            agg_res = online_func(agg_res, res, _step)

        agg_func = self.params['inference_params']['agg_func']
        val_result = agg_func(agg_res)
        
        return val_result


    # Testing functions
    def _run_test_loop(self):
        test_result = self.run_inference()
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
                    for v in self.trainable_variables()])
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
                tf.cast(self.global_step, tf.float32), 
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
                    x=self.global_step,
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
            k, val_num_clips,
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
                [-1, val_num_clips, num_classes])
        top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
        _, curr_pred = tf.nn.top_k(top_labels_one_hot, k=1)
        curr_pred = tf.squeeze(tf.cast(curr_pred, tf.int64), axis=1)
        imagenet_top1 = tf.reduce_mean(
                tf.cast(
                    tf.equal(curr_pred, inputs['y']),
                    tf.float32))
        return {'top1_{k}NN'.format(k=k): imagenet_top1}

