import os
import numpy as np
import yaml
import tensorflow as tf

import data
from model import instance_model
from utils.vie_utils import tuple_get_one


class ParamsLoader:
    def __init__(self, training_step):
        self.config = self._get_yml_file('settings/config.yml')
        self.args = self._get_yml_file('settings/args.yml')
        self.setting = training_step

        self._set_environment()

    def _get_yml_file(self, name):
        with open(name, 'r') as file:
            yml_file = yaml.safe_load(file)
        return yml_file


    def _set_environment(self):
        if not self.config['IS_GPU']:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    def _reg_loss(self, loss, weight_decay):
        # Add weight decay to the loss.
        l2_loss = weight_decay * tf.add_n(
                [tf.nn.l2_loss(tf.cast(v, tf.float32))
                    for v in tf.compat.v1.trainable_variables()])
        loss_all = tf.add(loss, l2_loss)
        return loss_all


    def _loss_func(self, output):
        return output['loss']


    def _get_lr_from_boundary_and_ramp_up(
            self,
            global_step, boundaries, 
            init_lr, target_lr, ramp_up_epoch,
            num_batches_per_epoch):
        curr_epoch  = tf.math.divide(
                tf.cast(global_step, tf.float32), 
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
                    x=global_step,
                    boundaries=boundaries, values=all_lrs)
        return curr_lr


    def _get_loss_lr_opt_params_from_arg(self, dataset_len):
        # loss_params: parameters to build the loss
        loss_params = {
            'pred_targets': [],
            'agg_func': self._reg_loss,
            'agg_func_kwargs': {'weight_decay': self.args['weight_decay']},
            'loss_func': self._loss_func,
        }

        # learning_rate_params: build the learning rate
        # For now, just stay the same
        learning_rate_params = {
                'func': self._get_lr_from_boundary_and_ramp_up,
                'init_lr': self.args['init_lr'],
                'target_lr': self.args['target_lr'] or self.args['init_lr'],
                'num_batches_per_epoch': dataset_len // self.args['batch_size'],
                'boundaries': self.args[self.setting]['lr_boundaries'],
                'ramp_up_epoch': self.args['ramp_up_epoch'],
                }

        # optimizer_params: use tfutils optimizer,
        # as mini batch is implemented there
        optimizer_params = {
                'optimizer': tf.compat.v1.train.MomentumOptimizer,
                'momentum': .9,
                }
        return loss_params, learning_rate_params, optimizer_params


    def _get_save_load_params_from_arg(self):
        # load_params: defining where to load, if needed
        load_exp_id = self.args[self.setting]['exp_id']
        train_steps = self.args['train_steps']
        
        # save_params: defining where to save the models
        fre_cache_filter = self.args['fre_cache_filter'] or self.args[self.setting]['fre_filter']
        dir_num_steps = ''
        for step in train_steps.split(','):
            dir_num_steps += str(self.args[step]['train_num_steps']) + '_'
        dir_num_steps = dir_num_steps[:-1]
        cache_dir = os.path.join(
                self.args['cache_dir'], dir_num_steps, load_exp_id)
        save_params = {
                'save_metrics_freq': self.args['fre_metric'],
                'save_valid_freq': self.args['fre_valid'],
                'save_filters_freq': self.args['fre_filter'],
                'cache_filters_freq': fre_cache_filter,
                'cache_dir': cache_dir,
                'train_log_file': self.args['train_log_file'],
                'val_log_file': self.args['val_log_file'],
                'test_log_file': self.args['test_log_file'],
                }
        
        load_exp = self.args[self.setting]['load_exp']
        load_step = None
        if load_exp:
            load_step = self.args[load_exp]['train_num_steps']
        load_query = None

        if not self.args['resume']:
            if load_exp is not None:
                load_exp_id = load_exp
            if load_step:
                load_query = {'exp_id': load_exp,
                              'saved_filters': True,
                              'step': load_step}

        load_params = {
                'exp_id': load_exp_id,
                'query': load_query}
        return save_params, load_params


    def _get_model_func_params(self, dataset_len):
        model_params = {
            "instance_t": self.args['instance_t'],
            "instance_k": self.args['instance_k'],
            "trn_use_mean": self.args['trn_use_mean'],
            "kmeans_k": self.args['kmeans_k'],
            "task": self.args[self.setting]['task'],
            "instance_data_len": dataset_len,
            "emb_dim": self.args['emb_dim']
        }
        return model_params


    def _rep_loss_func(
            self,
            inputs,
            output,
            gpu_offset=0,
            **kwargs
            ):
        data_indx = output['data_indx']
        new_data_memory = output['new_data_memory']
        loss_pure = output['loss']

        memory_bank_list = output['memory_bank']
        all_labels_list = output['all_labels']
        if isinstance(memory_bank_list, tf.Variable):
            memory_bank_list = [memory_bank_list]
            all_labels_list = [all_labels_list]

        devices = ['/gpu:%i' \
                % (idx + gpu_offset) for idx in range(len(memory_bank_list))]
        update_ops = []
        for device, memory_bank, all_labels \
                in zip(devices, memory_bank_list, all_labels_list):
            with tf.device(device):
                mb_update_op = tf.compat.v1.scatter_update(
                        memory_bank, data_indx, new_data_memory)
                update_ops.append(mb_update_op)
                lb_update_op = tf.compat.v1.scatter_update(
                        all_labels, data_indx,
                        inputs['label'])
                update_ops.append(lb_update_op)

        with tf.control_dependencies(update_ops):
            # Force the updates to happen before the next batch.
            loss_pure = tf.identity(loss_pure)

        ret_dict = {'loss_pure': loss_pure}
        for key, value in output.items():
            if key.startswith('loss_'):
                ret_dict[key] = value
        return ret_dict


    def _online_agg(self, agg_res, res, step):
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            agg_res[k].append(np.mean(v))
            # agg_res[k].append(v)
        return agg_res


    def _valid_perf_func_kNN(
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
                    tf.equal(curr_pred, inputs['label']),
                    tf.float32))
        return {'top1_{k}NN'.format(k=k): imagenet_top1}
    
    def _test_perf_func_kNN(
            self,
            inputs, output, 
            instance_t,
            k, num_classes,
            test_num_clips):
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
                [-1, test_num_clips, num_classes])
        top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
        _, curr_pred = tf.nn.top_k(top_labels_one_hot, k=1)
        curr_pred = tf.squeeze(tf.cast(curr_pred, tf.int64), axis=1)
        imagenet_top1 = tf.reduce_mean(
                tf.cast(
                    tf.equal(curr_pred, inputs['label']),
                    tf.float32))
        return {'top1_{k}NN'.format(k=k): imagenet_top1}

    def _valid_sup_func(
            self,
            inputs, output, 
            val_num_clips):
        num_classes = output.get_shape().as_list()[-1]
        curr_output = tf.nn.softmax(output)
        curr_output = tf.reshape(output, [-1, val_num_clips, num_classes])
        curr_output = tf.reduce_mean(curr_output, axis=1)

        top1_accuracy = tf.nn.in_top_k(curr_output, inputs['label'], k=1)
        top5_accuracy = tf.nn.in_top_k(curr_output, inputs['label'], k=5)
        return {'top1': top1_accuracy, 'top5': top5_accuracy}

    def _get_valid_loop_from_arg(self, val_data_loader):
        val_counter = [0]
        val_step_num = val_data_loader.dataset.__len__() // self.args['val_batch_size']
        val_data_enumerator = [enumerate(val_data_loader)]
        def valid_loop(sess, target):
            val_counter[0] += 1
            if val_counter[0] % val_step_num == 0:
                val_data_enumerator.pop()
                val_data_enumerator.append(enumerate(val_data_loader))
            _, (inputs, target) = next(val_data_enumerator[0])
            feed_dict = data.get_feeddict(inputs, target, name_prefix='VAL')
            return sess.run(target, feed_dict=feed_dict)
        return valid_loop, val_step_num    

    def _get_test_loop_from_arg(self, test_data_loader):
        test_counter = [0]
        test_step_num = test_data_loader.dataset.__len__() // self.args['test_batch_size']
        test_data_enumerator = [enumerate(test_data_loader)]
        def test_loop(sess, target):
            test_counter[0] += 1
            if test_counter[0] % test_step_num == 0:
                test_data_enumerator.pop()
                test_data_enumerator.append(enumerate(test_data_loader))
            _, (inputs, target) = next(test_data_enumerator[0])
            feed_dict = data.get_feeddict(inputs, target, name_prefix='TEST')
            return sess.run(target, feed_dict=feed_dict)
        return test_loop, test_step_num

    def _get_topn_val_data_param_from_arg(self):
        topn_val_data_param = {
                'func': data.get_placeholders,
                'batch_size': self.args['val_batch_size'],
                'num_frames': self.args['num_frames'] * self.args['val_num_clips'],
                'num_channels': self.args['input_shape']['num_channels'],
                'multi_frame': True,
                'multi_group': self.args['val_num_clips'],
                'name_prefix': 'VAL'}
        return topn_val_data_param
    
    def _get_topn_test_data_param_from_arg(self):
        topn_test_data_param = {
                'func': data.get_placeholders,
                'batch_size': self.args['test_batch_size'],
                'num_frames': self.args['num_frames'] * self.args['test_num_clips'],
                'num_channels': self.args['input_shape']['num_channels'],
                'multi_frame': True,
                'multi_group': self.args['test_num_clips'],
                'name_prefix': 'TEST'}
        return topn_test_data_param

    def _get_input_shape(self):
        num_channels = int(self.args['vgg_out_shape'])
        input_shape = {
            'num_channels': num_channels,
        }
        return input_shape

    def get_plot_params(self):
        save_params, _ = self._get_save_load_params_from_arg()
        plot_params = {
            'cache_dir': save_params['cache_dir'],
            'train_log_file': save_params['train_log_file'],
            'val_log_file': save_params['val_log_file'],
        }
        return plot_params


    def get_pie_params(self):
        pie_path = self.config['PIE_PATH']
        batch_size = self.args['batch_size']
        val_batch_size = self.args['val_batch_size']
        val_num_clips = self.args['val_num_clips']
        test_batch_size = self.args['test_batch_size']
        test_num_clips = self.args['test_num_clips']
        emb_dim = self.args['emb_dim']
        data_opts = {
            'fstride': 1,
            'sample_type': 'all', 
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',  #  kfold, random, default
            'seq_type': 'intention', #  crossing , intention
            'min_track_size': 0, #  discard tracks that are shorter
            'max_size_observe': self.args['num_frames'],  # number of observation frames
            'max_size_predict': 5,  # number of prediction frames
            'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
            'balance': True,  # balance the training and testing samples
            'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
            'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
            'encoder_input_type': [],
            'decoder_input_type': ['bbox'],
            'output_type': ['intention_binary']
        }
        pie_params = {
            'data_opts': data_opts,
            'pie_path': pie_path,
            'batch_size': batch_size,
            'val_batch_size': val_batch_size,
            'val_num_clips': val_num_clips,
            'test_batch_size': test_batch_size,
            'test_num_clips': test_num_clips,
            'emb_dim': emb_dim
        }
        return pie_params
    
    def get_test_params(self, data_loaders):
        test_data_loader = data_loaders['test']

        topn_test_data_param = self._get_topn_test_data_param_from_arg()

        test_targets = {
            'func': self._test_perf_func_kNN,
            'k': self.args['kNN_test'],
            'instance_t': self.args['instance_t'],
            'test_num_clips': self.args['test_num_clips'],
            'num_classes': self.args['num_classes'],
            }

        test_loop, test_step_num = self._get_test_loop_from_arg(test_data_loader)

        topn_test_param = {
            'data_params': topn_test_data_param,
            'queue_params': None,
            'targets': test_targets,
            'num_steps': test_step_num,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': self._online_agg,
            'test_loop': {'func': test_loop}
        }

        test_params = {
            'topn': topn_test_param,
        }

        params = {
            'test_params': test_params,
        }

        return params
    
    def _get_num_nodes(self, data_loader):
        _, (x, _, _) = next(enumerate(data_loader))
        print('x.shape', x.shape)
        return x.shape[1]

    def get_train_params(self, data_loaders, nn_clusterings):
        train_data_loader = data_loaders['train']
        val_data_loader = data_loaders['val']

        train_dataset_len = train_data_loader.dataset.__len__()

        num_nodes_per_graph = self._get_num_nodes(train_data_loader)
        loss_params, learning_rate_params, optimizer_params = self._get_loss_lr_opt_params_from_arg(train_dataset_len)

        first_step = []
        data_enumerator = [enumerate(train_data_loader)]
        def train_loop(sess, train_targets, num_minibatches=1, **params):
            assert num_minibatches==1, "Mini-batch not supported!"

            global_step_vars = [v for v in tf.compat.v1.global_variables() \
                                if 'global_step' in v.name]
            assert len(global_step_vars) == 1
            global_step = sess.run(global_step_vars[0])

            first_flag = len(first_step) == 0
            update_fre = self.args['clstr_update_fre'] or train_dataset_len // self.args['batch_size']
            if (global_step % update_fre == 0 or first_flag) \
                    and (nn_clusterings[0] is not None):
                if first_flag:
                    first_step.append(1)
                print("Recomputing clusters...")
                new_clust_labels = nn_clusterings[0].recompute_clusters(sess)
                for clustering in nn_clusterings:
                    clustering.apply_clusters(sess, new_clust_labels)

            if self.args['part_vd'] is None:
                data_en_update_fre = train_dataset_len // self.args['batch_size']
            else:
                new_length = int(train_dataset_len * self.args['part_vd'])
                data_en_update_fre = new_length // self.args['batch_size']

            # TODO: make this smart
            if global_step % data_en_update_fre == 0:
                data_enumerator.pop()
                data_enumerator.append(enumerate(train_data_loader))
            _, (inputs, y) = next(data_enumerator[0])
            x, a = inputs
            feed_dict = data.get_feeddict(x, a, y, name_prefix='TRAIN')
            sess_res = sess.run(train_targets, feed_dict=feed_dict)
            return sess_res

        # train_params: parameters about training data
        train_data_param = {
            'func': data.get_placeholders,
            'batch_size': self.args['batch_size'],
            'num_nodes': num_nodes_per_graph,
            'num_channels': self.args['input_shape']['num_channels'],
            'name_prefix': 'TRAIN'
        }
        train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': self.args[self.setting]['train_num_steps'],
            'train_loop': {'func': train_loop},
        }
        
        if not self.args[self.setting]['task'] == 'SUP':
            ## Add other loss reports (loss_model, loss_noise)
            train_params['targets'] = {
                'func': self._rep_loss_func
            }

        # validation_params: control the validation
        topn_val_data_param = self._get_topn_val_data_param_from_arg()

        if not self.args[self.setting]['task'] == 'SUP':
            val_targets = {
                    'func': self._valid_perf_func_kNN,
                    'k': self.args['kNN_val'],
                    'instance_t': self.args['instance_t'],
                    'val_num_clips': self.args['val_num_clips'],
                    'num_classes': self.args['num_classes']}
        else:
            val_targets = {
                    'func': self._valid_sup_func,
                    'val_num_clips': self.args['val_num_clips']}

        valid_loop, val_step_num = self._get_valid_loop_from_arg(val_data_loader)

        topn_val_param = {
            'data_params': topn_val_data_param,
            'queue_params': None,
            'targets': val_targets,
            'num_steps': val_step_num,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': self._online_agg,
            'valid_loop': {'func': valid_loop}
            }

        validation_params = {
            'topn': topn_val_param,
        }

        params = {
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'train_params': train_params,
            'validation_params': validation_params,
        }

        return params

    def get_params(self, data_loaders, is_test):
        save_params, load_params = self._get_save_load_params_from_arg()
        pie_params = self.get_pie_params()

        self.args['kmeans_k'] = [self.args['num_classes']]
        self.args['input_shape'] = self._get_input_shape()

        # model_params: a function that will build the model
        nn_clusterings = []
        def build_output(inputs, train, **kwargs):
            res = instance_model.build_output(inputs, train, **kwargs)
            if not train:
                return res
            outputs, logged_cfg, clustering = res
            nn_clusterings.append(clustering)
            return outputs, logged_cfg

        model_func_params = self._get_model_func_params(data_loaders['train'].dataset.__len__())
        model_params = {
            'func': build_output,
            'model_func_params': model_func_params
        }

        gen_params = {
            'is_test': is_test,
            'pie_path': pie_params['pie_path'],
            'save_params': save_params,
            'load_params': load_params,
            'data_opts': pie_params['data_opts'],
            'model_params': model_params,
        }
        train_params = self.get_train_params(data_loaders, nn_clusterings)

        test_params = {}
        if is_test:
            test_params = self.get_test_params(data_loaders)

        all_params = {**gen_params, **train_params, **test_params}
        return all_params