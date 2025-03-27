import os
import yaml

class ParamsLoader:
    def __init__(self, training_step, num_kfolds, fold):
        self.config = self._get_yml_file('settings/config.yml')
        self.args = self._get_yml_file('settings/args.yml')
        self.setting = training_step
        self.num_kfolds = num_kfolds
        self.fold = fold

        self._set_environment()

    @staticmethod
    def get_dataset_params_static(args, config, num_kfolds, fold):
        data_opts = {
            'fstride': 1,
            'sample_type': 'all', 
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'kfold' if num_kfolds else 'default',  #  kfold, random, default,
            'seq_type': 'intention', #  crossing , intention
            'min_track_size': args['num_frames'], #  discard tracks that are shorter
            'max_size_observe': args['num_frames'],  # number of observation frames
            'max_size_predict': 5,  # number of prediction frames
            'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
            'balance': True,  # balance the training and testing samples
            'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
            'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
            'encoder_input_type': [],
            'decoder_input_type': ['bbox'],
            'output_type': ['intention_binary'],
            'random_params': {'ratios': None,
                              'val_data': True,
                              'regen_data': False},
            'kfold_params': {'num_folds': num_kfolds, 'fold': fold},
        }
        dataset_params = {
            'data_opts': data_opts,
            'psi_path': config['PSI_PATH'],
            'pie_path': config['PIE_PATH'],
            'batch_size': args['batch_size'],
            'inference_batch_size': args['inference_batch_size'],
            'img_height': args['img_height'],
            'img_width': args['img_width'],
            'edge_weigths': args['edge_weigths'],
            'edge_importance': args['edge_importance'],
            'feature_extractor': args['feature_extractor'],
            'feat_input_size': args[args['feature_extractor']+'_input_size'],
            'data_sets': args['data_sets'],
            'balance_dataset': args['balance_dataset'],
            'obj_classes': [] if args['obj_classes'] == '' else args['obj_classes'].split(','),
        }
        return dataset_params
    

    def get_dataset_params(self):
        return self.get_dataset_params_static(self.args, self.config, self.num_kfolds, self.fold)


    def get_args(self, dataset):
        self.args['dataset_name'] = dataset
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
            'agg_func_kwargs': {'weight_decay': self.args[self.setting]['weight_decay']},
        }

        # learning_rate_params: build the learning rate
        # For now, just stay the same
        train_num_epochs = self.args[self.setting]['train_num_epochs']
        init_lr = self.args[self.setting]['init_lr']
        target_lr = self.args[self.setting]['target_lr']
        lr_decay_start = self.args[self.setting]['lr_decay_start']
        lr_decay = self.args[self.setting]['lr_decay']
        lr_decay_steps = self.args[self.setting]['lr_decay_steps']
        boundaries = list(range(lr_decay_start, train_num_epochs, lr_decay_steps))
        learning_rates = [init_lr - (lr_decay * i) if init_lr - (lr_decay * i) > target_lr else target_lr \
                           for i in range(len(boundaries) + 1)]
        learning_rate_params = {
            'learning_rates': learning_rates,
            'boundaries': boundaries,
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
        if self.num_kfolds:
            dir_num_epochs += 'kfold_' + str(self.num_kfolds) + '_' + str(self.fold) + '_'
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

    def _get_model_func_params(self, train_dataset_len):
        model_params = {
            "instance_t": self.args['instance_t'],
            "instance_k": self.args[self.setting]['instance_k'],
            "kmeans_k": self.args['kmeans_k'],
            "task": self.args[self.setting]['task'],
            "data_len": train_dataset_len,
            "emb_dim": self.args['emb_dim'],
            "gcn_middle_layer_dim": self.args['gcn_middle_layer_dim'],
            "gcn_middle_2_layer_dim": self.args['gcn_middle_2_layer_dim'],
            "gcn_input_layer_dim": self.args['gcn_input_layer_dim'],
            "gcn_output_layer_dim": self.args['gcn_output_layer_dim'],
            "drop_conv": self.args['drop_conv'],
            "drop_tcn": self.args['drop_tcn'],
            "drop_lstm": self.args['drop_lstm'],
            "gcn_num_input_layers": self.args['gcn_num_input_layers'],
            "gcn_num_middle_layers": self.args['gcn_num_middle_layers'],
            "gcn_num_middle_2_layers": self.args['gcn_num_middle_2_layers'],
            "gcn_num_output_layers": self.args['gcn_num_output_layers'],
            "scene_input_layer_dim": self.args['scene_input_layer_dim'],
            "scene_output_layer_dim": self.args['scene_output_layer_dim'],
            "scene_num_input_layers": self.args['scene_num_input_layers'],
            "scene_num_output_layers": self.args['scene_num_output_layers'],
            "seq_len": self.args['num_frames'],
            "num_nodes": len([] if self.args['obj_classes'] == '' else self.args['obj_classes'].split(',')) + 1, # ped + obj_classes
            "edge_importance": self.args['edge_importance'],
            "is_scene": self.args['is_scene'],
            "share_edge_importance": self.args['share_edge_importance'],
            "cluster_alg": self.args['cluster_alg'],
            "num_classes": self.args['num_classes'],
            "stgcn_kernel_size": self.args['stgcn_kernel_size'],
            "feat_output_size": self.args[self.args['feature_extractor']+'_output_size'],
            "batch_size": self.args['batch_size'],
        }
        return model_params

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

        params = {
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'train_params': train_params,
        }

        return params
    
    def get_inference_params(self):
        inference_targets = {
            'k': self.args['kNN_inference'],
            'instance_t': self.args['instance_t'],
            'num_classes': self.args['num_classes']
        }
        inference_params = {
            'targets': inference_targets,
            'inference_batch_size': self.args['inference_batch_size'],
        }
        params = {
            'inference_params': inference_params,
        }
        return params

    def get_params(self, datasets, is_test, is_only_test):

        save_params, load_params = self._get_save_load_params_from_arg()
        pie_params = self.get_dataset_params()

        self.args['emb_dim'] = self.args['scene_output_layer_dim'] + self.args['gcn_output_layer_dim']
        self.args['kmeans_k'] = [self.args['clustering_groups']] * self.args['num_kmeans']
        model_func_params = self._get_model_func_params(datasets['train']['len'] if not is_only_test else None) # datasets['train']['len'] only set when training
        model_params = {
            'model_func_params': model_func_params
        }

        generic_params = {
            'is_test': is_test,
            'is_only_test': is_only_test,
            'save_params': save_params,
            'load_params': load_params,
            'data_opts': pie_params['data_opts'],
            'model_params': model_params,
            'datasets': datasets,
        }
        train_params = self.get_train_params(datasets) if not is_only_test else {} # datasets only set when training
        inference_params = self.get_inference_params()

        all_params = {**generic_params, **train_params, **inference_params}
        return all_params