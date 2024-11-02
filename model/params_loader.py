import os
import numpy as np
import yaml
import tensorflow as tf

import data
from model import instance_model


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


    def _get_loss_lr_opt_params_from_arg(self, dataset_len):
        # loss_params: parameters to build the loss
        loss_params = {
            'agg_func_kwargs': {'weight_decay': self.args['weight_decay']},
        }

        # learning_rate_params: build the learning rate
        # For now, just stay the same
        learning_rate_params = {
            'init_lr': self.args['init_lr'],
            'target_lr': self.args['target_lr'] or self.args['init_lr'],
            'num_batches_per_epoch': dataset_len // self.args['batch_size'],
            'boundaries': self.args[self.setting]['lr_boundaries'],
            'ramp_up_epoch': self.args['ramp_up_epoch'],
        }

        # optimizer_params: use tfutils optimizer,
        # as mini batch is implemented there
        optimizer_params = {
            'momentum': .9,
        }
        return loss_params, learning_rate_params, optimizer_params


    def _get_save_load_params_from_arg(self):
        # load_params: defining where to load, if needed
        load_exp_id = self.args[self.setting]['exp_id']
        train_steps = self.args['train_steps']
        
        # save_params: defining where to save the models
        dir_num_steps = ''
        for step in train_steps.split(','):
            dir_num_steps += str(self.args[step]['train_num_steps']) + '_'
        dir_num_steps = dir_num_steps[:-1]
        cache_dir = os.path.join(
                self.args['cache_dir'], dir_num_steps, load_exp_id)
        save_params = {
                'save_metrics_freq': self.args['fre_metric'],
                'save_valid_freq': self.args['fre_valid'],
                'fre_save_model': self.args['fre_save_model'],
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
            "emb_dim": self.args['emb_dim'],
            "middle_dim": self.args['middle_dim'],
        }
        return model_params


    def _online_agg(self, agg_res, res, step):
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            agg_res[k].append(np.mean(v))
            # agg_res[k].append(v)
        return agg_res

    def _get_inference_loop_from_arg(self, data_loader):
        counter = [0]
        step_num = data_loader.dataset.__len__() // self.args['inference_batch_size']
        data_enumerator = [enumerate(data_loader)]
        def inference_loop(inference_func):
            counter[0] += 1
            if counter[0] % step_num == 0:
                data_enumerator.pop()
                data_enumerator.append(enumerate(data_loader))
            _, (x, a, y, i) = next(data_enumerator[0])
            inputs = {
                'x': x,
                'a': a,
                'y': y,
                'i': i,
            }
            res = inference_func(inputs, train=False)
            return res
        return inference_loop, step_num    

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
        img_height = self.args['img_height']
        img_width = self.args['img_width']
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
            'emb_dim': emb_dim,
            'img_height': img_height,
            'img_width': img_width,
        }
        return pie_params
    
    def get_test_params(self, data_loaders):
        test_data_loader = data_loaders['test']

        test_targets = {
            'k': self.args['kNN_test'],
            'instance_t': self.args['instance_t'],
            'test_num_clips': self.args['test_num_clips'],
            'num_classes': self.args['num_classes'],
            }

        test_loop, test_step_num = self._get_inference_loop_from_arg(test_data_loader)

        test_params = {
            'queue_params': None,
            'targets': test_targets,
            'num_steps': test_step_num,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': self._online_agg,
            'test_loop': {'func': test_loop}
        }

        params = {
            'test_params': test_params,
        }

        return params
    
    def _get_num_nodes(self, data_loader):
        _, (x, _, _, _) = next(enumerate(data_loader))
        return x.shape[2]

    def get_train_params(self, data_loaders, nn_clusterings):
        train_data_loader = data_loaders['train']
        val_data_loader = data_loaders['val']

        train_dataset_len = train_data_loader.dataset.__len__()

        loss_params, learning_rate_params, optimizer_params = self._get_loss_lr_opt_params_from_arg(train_dataset_len)

        first_step = []
        data_enumerator = [enumerate(train_data_loader)]
        def train_loop(train_function, num_minibatches=1, **params):
            assert num_minibatches==1, "Mini-batch not supported!"

            global_step_vars = [v for v in tf.compat.v1.global_variables() \
                                if 'global_step' in v.name]
            assert len(global_step_vars) == 1
            global_step = global_step_vars[0]

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
            _, (x, a, y, i) = next(data_enumerator[0])
            inputs = {
                'x': x,
                'a': a,
                'y': y,
                'i': i,
            }
            res = train_function(inputs, train=True)
            return res

        train_params = {
            'validate_first': False,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': self.args[self.setting]['train_num_steps'],
            'train_loop': {'func': train_loop},
        }
        
        if not self.args[self.setting]['task'] == 'SUP':
            ## Add other loss reports (loss_model, loss_noise)
            train_params['targets'] = {}

        valid_loop, val_step_num = self._get_inference_loop_from_arg(val_data_loader)

        inference_targets = {
            'k': self.args['kNN_inference'],
            'instance_t': self.args['instance_t'],
            'inference_num_clips': self.args['inference_num_clips'],
            'num_classes': self.args['num_classes']
        }

        inference_params = {
            'queue_params': None,
            'targets': inference_targets,
            'num_steps': val_step_num,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': self._online_agg,
            'valid_loop': {'func': valid_loop}
        }

        params = {
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'train_params': train_params,
            'inference_params': inference_params,
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