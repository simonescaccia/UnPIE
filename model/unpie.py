import os
import time
import tqdm
import numpy as np
import tensorflow as tf

from utils.print_utils import print_separator

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

    def run_each_validation(self, val_key):
        agg_res = None
        num_steps = self.params['validation_params'][val_key]['num_steps']
        for _step in tqdm.trange(num_steps, desc=val_key):
            if self.params['validation_params'][val_key].get('valid_loop', None) is None:
                res = self.sess.run(self.all_val_targets[val_key])
            else:
                res = self.params['validation_params'][val_key]['valid_loop']['func'](
                        self.sess, self.all_val_targets[val_key])
            online_func = self.params['validation_params'][val_key]['online_agg_func']
            agg_res = online_func(agg_res, res, _step)
        agg_func = self.params['validation_params'][val_key]['agg_func']
        val_result = agg_func(agg_res)
        return val_result
    
    def run_testing(self, test_key):
        agg_res = None
        num_steps = self.params['test_params'][test_key]['num_steps']
        for _step in tqdm.trange(num_steps, desc=test_key):
            if self.params['test_params'][test_key].get('test_loop', None) is None:
                res = self.sess.run(self.all_test_targets[test_key])
            else:
                res = self.params['test_params'][test_key]['test_loop']['func'](
                        self.sess, self.all_test_targets[test_key])
            online_func = self.params['test_params'][test_key]['online_agg_func']
            agg_res = online_func(agg_res, res, _step)
        agg_func = self.params['test_params'][test_key]['agg_func']
        test_result = agg_func(agg_res)
        return test_result

    def train_func(self, inputs, train):
        outputs = self.build_network(inputs, train=True)
        loss_retval, learning_rate, train_opt = self.build_train_opt(outputs)     
        train_targets = self.build_train_targets(outputs, loss_retval, learning_rate, train_opt)
        
        return train_targets

    def run_train_loop(self):
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
                for each_val_key in self.params['validation_params']:
                    val_result = self.run_each_validation(each_val_key)
                    self.val_log_writer.write(
                            '%s: %s\n' % (each_val_key, str(val_result)))
                    print(val_result)
                self.val_log_writer.close()
                self.val_log_writer = open(self.val_log_file_path, 'a+')

    def run_test_loop(self):
        for each_val_key in self.params['test_params']:
            test_result = self.run_testing(each_val_key)
            self.test_log_writer.write(
                    '%s: %s\n' % (each_val_key, str(test_result)))
            print(test_result)
        self.test_log_writer.close()

    def build_val_inputs(self, val_key):
        data_params = self.params['validation_params'][val_key]['data_params']
        func = data_params.pop('func')
        val_inputs = func(**data_params)
        return val_inputs

    def build_test_inputs(self, test_key):
        data_params = self.params['test_params'][test_key]['data_params']
        func = data_params.pop('func')
        test_inputs = func(**data_params)
        return test_inputs

    def build_val_network(self, val_key, val_inputs):
        with tf.name_scope('validation/' + val_key):
            val_outputs = self.build_network(val_inputs, False)
        return val_outputs
    
    def build_test_network(self, test_key, test_inputs):
        with tf.name_scope('test/' + test_key):
            test_outputs = self.build_network(test_inputs, False)
        return test_outputs

    def build_val_targets(self, val_key, val_inputs, val_outputs):
        target_params = self.params['validation_params'][val_key]['targets']
        func = target_params.pop('func')
        val_targets = func(val_inputs, val_outputs, **target_params)
        return val_targets
    
    def build_test_targets(self, test_key, test_inputs, test_outputs):
        target_params = self.params['test_params'][test_key]['targets']
        func = target_params.pop('func')
        test_targets = func(test_inputs, test_outputs, **target_params)
        return test_targets

    def build_val(self):
        tf.compat.v1.get_variable_scope().reuse_variables()
        self.all_val_targets = {}
        for each_val_key in self.params['validation_params']:
            val_inputs = self.build_val_inputs(each_val_key)
            val_outputs = self.build_val_network(each_val_key, val_inputs)
            val_targets = self.build_val_targets(
                    each_val_key, val_inputs, val_outputs)
            self.all_val_targets[each_val_key] = val_targets

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
        self.build_train()
        self.build_val()
        if self.params['is_test']:
            self.build_test()

        self.build_sess_and_saver()
        self.init_and_restore()


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

    # Running functions
    def train(self):
        print_separator('Starting UnPIE training')
        self.run_train_loop()
        print_separator('UnPIE training ended')

    def test(self):
        print_separator('Starting UnPIE testing')
        self.run_test_loop()
        print_separator('UnPIE testing ended')
