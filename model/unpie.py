import os
import time
import numpy as np

import tensorflow as tf

from dataset.pie_data import PIE
from model.temporal_aggregator import TemporalAggregator
from utils.print_utils import print_separator


class UnPIE(object):
    def __init__(self, params):
        self.params = params

        self.temporal_aggregator = TemporalAggregator()

        self.cache_dir = self.params['save_params']['cache_dir'] # Set cache directory
        self.log_file_path = os.path.join(self.cache_dir, 'log.txt')
        self.val_log_file_path = os.path.join(self.cache_dir, 'val_log.txt')
        os.system('mkdir -p %s' % self.cache_dir)
        self.load_from_curr_exp = tf.train.latest_checkpoint(self.cache_dir)
        if not self.load_from_curr_exp: # if no checkpoint is found then create a new log file
            self.log_writer = open(self.log_file_path, 'w')
            self.val_log_writer = open(self.val_log_file_path, 'w')
        else: # if checkpoint is found then append to the existing log file
            self.log_writer = open(self.log_file_path, 'a+')
            self.val_log_writer = open(self.val_log_file_path, 'a+')

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)     

    def build_train_val_data(self):
        self.build_inputs()
        self.build_network(self.inputs, train=True)
    
    def build_inputs(self):
        data_params = self.params['train_params']['data_params']
        func = data_params.pop('func')
        self.inputs = func(**data_params)

    def build_network(self, inputs, train):
        self.params['model_params']['model_func_params']['instance_data_len'] = self.train_img.shape[0]
        model_params = self.params['model_params']
        model_func_params = model_params['model_func_params']
        func = model_params.pop('func')
        self.outputs, _ = func(
                inputs=inputs,
                train=train,
                **model_func_params)

    def build_train_op(self):
        self.params['learning_rate_params']['num_batches_per_epoch'] = \
            self.train_img.shape[0] // self.params['train_params']['data_params']['batch_size']
        loss_params = self.params['loss_params']

        input_targets = [self.inputs[key] \
                for key in loss_params['pred_targets']]
        func = loss_params['loss_func']
        self.loss_retval = func(
                self.outputs, 
                *input_targets, 
                **loss_params.get('loss_func_kwargs', {}))
        self.loss_retval = loss_params['agg_func'](
                self.loss_retval,
                **loss_params.get('agg_func_kwargs', {}))

        self.global_step = tf.compat.v1.get_variable(
                'global_step', [],
                dtype=tf.int64, trainable=False,
                initializer=tf.constant_initializer(0))
        lr_rate_params = self.params['learning_rate_params']
        func = lr_rate_params.pop('func')
        learning_rate = func(self.global_step, **lr_rate_params)
        self.learning_rate = learning_rate

        opt_params = self.params['optimizer_params']
        func = opt_params.pop('optimizer')
        opt = func(learning_rate=learning_rate, **opt_params)

        with tf.control_dependencies(
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.train_op = opt.minimize(
                    self.loss_retval, 
                    global_step=self.global_step)

    def load_from_ckpt(self, ckpt_path):
        print('Restore from %s' % ckpt_path)
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
            split_cache_path = self.cache_dir.split('/')
            split_cache_path[-1] = self.params['load_params']['exp_id']
            split_cache_path[-2] = self.params['load_params']['collname']
            split_cache_path[-3] = self.params['load_params']['dbname']
            load_dir = '/'.join(split_cache_path)
            if self.params['load_params']['query']:
                ckpt_path = os.path.join(
                        load_dir, 
                        'model.ckpt-%i' % self.params['load_params']['query']['step'])
            else:
                ckpt_path = tf.train.latest_checkpoint(load_dir)
            if ckpt_path:
                print('Restore from %s' % ckpt_path)
                #self.load_from_ckpt(ckpt_path)
                reader = tf.train.NewCheckpointReader(ckpt_path)
                saved_var_shapes = reader.get_variable_to_shape_map()

                all_vars = tf.global_variables()
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
                _load_saver = tf.train.Saver(var_list=filtered_var_list)
                _load_saver.restore(self.sess, ckpt_path)

    def training_loop(self):
        pass

    def run_train_loop(self):
        start_step = self.sess.run(self.global_step)
        train_loop = self.train_params.get('train_loop', None)

        import sys
        sys.exit(0)
        for curr_step in range(start_step, int(self.train_params['num_steps']+1)):
            self.start_time = time.time()
            if train_loop is None:
                train_res = self.sess.run(self.train_targets)
            else:
                train_res = train_loop['func'](self.sess, self.train_targets)

            duration = time.time() - self.start_time

            message = 'Step {} ({:.0f} ms) -- '\
                    .format(curr_step, 1000 * duration)
            rep_msg = ['{}: {:.4f}'.format(k, v) \
                    for k, v in train_res.items()
                    if k != 'train_op']
            message += ', '.join(rep_msg)
            print(message)

            if curr_step % self.save_params['cache_filters_freq'] == 0 \
                    and curr_step > 0:
                print('Saving model...')
                self.saver.save(
                        self.sess, 
                        os.path.join(
                            self.cache_dir,
                            'model.ckpt'), 
                        global_step=curr_step)

            self.log_writer.write(message + '\n')
            if curr_step % self.save_params['save_metrics_freq'] == 0:
                self.log_writer.close()
                self.log_writer = open(self.log_file_path, 'a+')

            if curr_step % self.save_params['save_valid_freq'] == 0:
                for each_val_key in self.validation_params:
                    val_result = self.run_each_validation(each_val_key)
                    self.val_log_writer.write(
                            '%s: %s\n' % (each_val_key, str(val_result)))
                    print(val_result)
                self.val_log_writer.close()
                self.val_log_writer = open(self.val_log_file_path, 'a+')

    def train(self):
        print_separator('Starting UnPIE training')

        self.build_train_val_data()
        self.build_train_op()

        self.build_sess_and_saver()
        self.init_and_restore()

        self.run_train_loop()
