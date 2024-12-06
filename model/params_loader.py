import os
import yaml

class ParamsLoader:
    def __init__(self, training_step):
        self.config = self._get_yml_file('settings/config.yml')
        self.args = self._get_yml_file('settings/args.yml')
        self.setting = training_step

        self._set_environment()

    @staticmethod
    def get_pie_params_static(args, config):
        data_opts = {
            'fstride': 1,
            'sample_type': 'all', 
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',  #  kfold, random, default
            'seq_type': 'intention', #  crossing , intention
            'min_track_size': 0, #  discard tracks that are shorter
            'max_size_observe': args['num_frames'],  # number of observation frames
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
            'pie_path': config['PIE_PATH'],
            'batch_size': args['batch_size'],
            'inference_batch_size': args['inference_batch_size'],
            'inference_num_clips': args['inference_num_clips'],
            'img_height': args['img_height'],
            'img_width': args['img_width'],
            'edge_weigths': args['edge_weigths'],
            'edge_importance': args['edge_importance'],
            'feature_extractor': args['feature_extractor'],
            'data_sets': args['data_sets'],
        }
        return pie_params


    def get_pie_params(self):
        return self.get_pie_params_static(self.args, self.config)


    def get_args(self):
        return self.args


    def _get_yml_file(self, name):
        with open(name, 'r') as file:
            yml_file = yaml.safe_load(file)
        return yml_file


    def _set_environment(self):
        if not self.config['IS_GPU']:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


    def _get_loss_lr_opt_params_from_arg(self, dataset_len):
        # loss_params: parameters to build the loss
        loss_params = {
            'agg_func_kwargs': {'weight_decay': self.args['weight_decay']},
        }

        # learning_rate_params: build the learning rate
        # For now, just stay the same
        lrs = self.args['learning_rates']
        b = self.args[self.setting]['lr_boundaries']
        learning_rate_params = {
            'learning_rates': [float(rate) for rate in lrs.split(",")],
            'boundaries': [int(boundary) for boundary in b.split(',')] if b else None,
            'steps_per_epoch': (dataset_len // self.args['batch_size']) + (dataset_len % self.args['batch_size'] > 0),
        }

        # optimizer_params: use tfutils optimizer,
        # as mini batch is implemented there
        optimizer_params = {
            'momentum': .9,
        }
        return loss_params, learning_rate_params, optimizer_params


    def _get_save_load_params_from_arg(self):
        # load_params: defining where to load, if needed
        task = self.args[self.setting]['task']
        train_steps = self.args['train_steps']
        
        # save_params: defining where to save the models
        dir_num_epochs = ''
        for step in train_steps.split(','):
            dir_num_epochs += str(self.args[step]['train_num_epochs']) + '_'
        dir_num_epochs = dir_num_epochs[:-1]
        cache_dir = os.path.join(
                self.args['cache_dir'], dir_num_epochs, task)
        save_params = {
            'save_valid_freq': self.args['fre_valid'],
            'fre_save_model': self.args['fre_save_model'],
            'cache_dir': cache_dir,
            'train_log_file': self.args['train_log_file'],
            'val_log_file': self.args['val_log_file'],
            'val_prediction_file': self.args['val_prediction_file'],
            'test_log_file': self.args['test_log_file'],
            'plot_dir': self.args['plot_dir'],
        }
        
        load_task = self.args[self.setting]['load_task']
        load_step = None
        if load_task:
            task = load_task
            load_step = self.args[load_task]['train_num_epochs']
  
        load_params = None
        if load_step:
            load_params = {
                'task': task,
                'step': load_step
            }

        return save_params, load_params

    def _get_model_func_params(self, dataset_len, num_nodes):
        model_params = {
            "instance_t": self.args['instance_t'],
            "instance_k": self.args[self.setting]['instance_k'],
            "kmeans_k": self.args['kmeans_k'],
            "task": self.args[self.setting]['task'],
            "data_len": dataset_len,
            "emb_dim": self.args['emb_dim'],
            "middle_dim": self.args['middle_dim'],
            "middle_2_dim": self.args['middle_2_dim'],
            "input_dim": self.args['channel_dim'],
            "gcn_dim": self.args['gcn_dim'],
            "drop_conv": self.args['drop_conv'],
            "drop_tcn": self.args['drop_tcn'],
            "num_input_layers": self.args['num_input_layers'],
            "num_middle_layers": self.args['num_middle_layers'],
            "num_middle_2_layers": self.args['num_middle_2_layers'],
            "num_gcn_final_layers": self.args['num_gcn_final_layers'],
            "scene_gcn_dim": self.args['scene_gcn_dim'],
            "scene_layers": self.args['scene_layers'],
            "seq_len": self.args['num_frames'],
            "num_nodes": num_nodes,
            "edge_importance": self.args['edge_importance'],
            "is_scene": self.args['is_scene'],
            "share_edge_importance": self.args['share_edge_importance'],
            "cluster_alg": self.args['cluster_alg'],
        }
        return model_params

    def _get_input_shape(self):
        num_channels = self.args['channel_dim']
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
            'plot_dir': self.args['plot_dir'],
        }
        return plot_params

    def get_train_params(self, datasets):
        train_dataset_len = datasets['train']['len']

        train_num_steps = (train_dataset_len // self.args['batch_size']) + (train_dataset_len % self.args['batch_size'] > 0)

        loss_params, learning_rate_params, optimizer_params = self._get_loss_lr_opt_params_from_arg(train_dataset_len)       

        train_params = {
            'validate_first': False,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_epochs': self.args[self.setting]['train_num_epochs'],
            'num_steps': train_num_steps,
            'clstr_update_per_epoch': self.args[self.setting]['clstr_update_per_epoch'],
            'fre_plot_clusters': self.args['fre_plot_clusters'],
        }
        
        if not self.args[self.setting]['task'] == 'SUP':
            ## Add other loss reports (loss_model, loss_noise)
            train_params['targets'] = {}

        inference_targets = {
            'k': self.args['kNN_inference'],
            'instance_t': self.args['instance_t'],
            'inference_num_clips': self.args['inference_num_clips'],
            'num_classes': self.args['num_classes']
        }

        inference_params = {
            'targets': inference_targets,
            'inference_batch_size': self.args['inference_batch_size'],
        }

        params = {
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'train_params': train_params,
            'inference_params': inference_params,
        }

        return params

    def get_params(self, datasets, is_test):
        save_params, load_params = self._get_save_load_params_from_arg()
        pie_params = self.get_pie_params()

        self.args['emb_dim'] = self.args['scene_gcn_dim'] + self.args['gcn_dim']
        self.args['kmeans_k'] = [self.args['num_classes']] * self.args['num_kmeans']
        self.args['input_shape'] = self._get_input_shape()

        model_func_params = self._get_model_func_params(datasets['train']['len'], datasets['train']['num_nodes'])
        model_params = {
            'model_func_params': model_func_params
        }

        generic_params = {
            'is_test': is_test,
            'pie_path': pie_params['pie_path'],
            'save_params': save_params,
            'load_params': load_params,
            'data_opts': pie_params['data_opts'],
            'model_params': model_params,
            'datasets': datasets,
        }
        train_params = self.get_train_params(datasets)

        all_params = {**generic_params, **train_params}
        return all_params