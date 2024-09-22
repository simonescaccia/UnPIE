import os
import pickle5 as pickle
# import pickle (tensorflow 2)
import time
import numpy as np
from pathlib import PurePath

import tensorflow as tf

from dataset.pie_data import PIE
from model.temporal_aggregator import TemporalAggregator
from utils.pie_utils import update_progress


class UnPIE(object):
    def __init__(self, params):
        self.params = params
        self.data_opts = params['data_opts']

        self.pie = PIE(data_path=params['pie_path'])
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

    def get_path(self,
                 type_save='models', # model or data
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder=''):
        """
        A path generator method for saving model and config data. Creates directories
        as needed.
        :param type_save: Specifies whether data or model is saved.
        :param models_save_folder: model name (e.g. train function uses timestring "%d%b%Y-%Hh%Mm%Ss")
        :param model_name: model name (either trained convlstm_encdec model or vgg16)
        :param file_name: Actual file of the file (e.g. model.h5, history.h5, config.pkl)
        :param data_subset: train, test or val
        :param data_type: type of the data (e.g. features_context_pad_resize)
        :param save_root_folder: The root folder for saved data.
        :return: The full path for the save folder
        """
        if save_root_folder == '':
            save_root_folder =  os.path.join(self.params['pie_path'], 'data')
        assert(type_save in ['models', 'data'])
        if data_type != '':
            assert(any([d in data_type for d in ['images', 'features']]))
        root = os.path.join(save_root_folder, type_save)

        if type_save == 'models':
            save_path = os.path.join(save_root_folder, 'pie', 'intention', models_save_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return os.path.join(save_path, file_name), save_path
        else:
            save_path = os.path.join(root, 'pie', data_subset, data_type, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return save_path

    def load_features(self,
                      img_sequences,
                      bbox_sequences,
                      ped_ids,
                      load_path,
                      data_type='train'):
        """
        Load image features. The images are first
        cropped to 1.5x the size of the bounding box, padded and resized to
        (224, 224) and fed into pretrained VGG16.
        :param img_sequences: a list of frame names
        :param bbox_sequences: a list of corresponding bounding boxes
        :ped_ids: a list of pedestrian ids associated with the sequences
        :save_path: path to save the precomputed features
        :data_type: train/val/test data set
        :regen_pkl: if set to True overwrites previously saved features
        :return: a list of image features
        """
        # load the feature files if exists
        print("Loading {} features crop_type=context crop_mode=pad_resize \nsave_path={}, ".format(data_type, load_path))

        sequences = []
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_sequences[i], pid):
                set_id = PurePath(imp).parts[-3]
                vid_id = PurePath(imp).parts[-2]
                img_name = PurePath(imp).parts[-1].split('.')[0]
                img_save_folder = os.path.join(load_path, set_id, vid_id)
                img_save_path = os.path.join(img_save_folder, img_name+'_'+p[0]+'.pkl')
                if not os.path.exists(img_save_path):
                    Exception("Image features not found at {}".format(img_save_path))
                with open(img_save_path, 'rb') as fid:
                    try:
                        img_features = pickle.load(fid)
                    except:
                        img_features = pickle.load(fid, encoding='bytes')
                img_features = np.squeeze(img_features) # VGG16 output shape: (7, 7, 512)
                img_seq.append(img_features)
            sequences.append(img_seq)
        update_progress(1)
        print("\n")
        sequences = np.array(sequences)
        return sequences

    def get_tracks(self, sequences, seq_length, overlap_stride):
        """
        Generate tracks by sampling from pedestrian sequences
        :param dataset: raw data from the dataset
        """
        sub_seqs = []
        for seq in sequences:
            sub_seqs.extend([seq[i:i+seq_length] for i in\
                         range(0,len(seq)\
                        - seq_length + 1, overlap_stride)])
        return sub_seqs

    def get_train_val_data(self, dataset, seq_length, overlap):
        """
        A helper function for data generation that combines different data types into a single
        representation.
        :param data: A dictionary of data types
        :param seq_length: the length of the sequence
        :param overlap: defines the overlap between consecutive sequences (between 0 and 1)
        :return: A unified data representation as a list.
        """
        bboxes = dataset['bbox'].copy() # shape: (num_ped, num_frames, 4)
        images = dataset['image'].copy() # shape: (num_ped, num_frames, 68)
        ped_ids = dataset['ped_id'].copy() # shape: (num_ped, num_frames, 1)
        int_bin = dataset['intention_binary'].copy() # shape: (num_ped, num_frames, 1)

        overlap_stride = seq_length if overlap == 0 else \
        int((1 - overlap) * seq_length)

        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        bboxes = self.get_tracks(bboxes, seq_length, overlap_stride)
        images = self.get_tracks(images, seq_length, overlap_stride)
        ped_ids = self.get_tracks(ped_ids, seq_length, overlap_stride)
        int_bin = self.get_tracks(int_bin, seq_length, overlap_stride)

        int_bin = np.array(int_bin)[:, 0] # every frame has the same intention label

        return {'images': images,
                'bboxes': bboxes,
                'ped_ids': ped_ids,
                'output': int_bin}

    def build_train_val_data(self):
        '''
        Build the inputs for the clustering computation
        '''
        # Generate image sequences
        seq_train = self.pie.generate_data_trajectory_sequence('train', **self.data_opts)
        seq_train = self.pie.balance_samples_count(seq_train, label_type='intention_binary')

        seq_val = self.pie.generate_data_trajectory_sequence('val', **self.data_opts)
        seq_val = self.pie.balance_samples_count(seq_val, label_type='intention_binary')
        
        seq_length = self.data_opts['max_size_observe']
        seq_ovelap_rate = self.data_opts['seq_overlap_rate']
        train_d = self.get_train_val_data(seq_train, seq_length, seq_ovelap_rate)
        val_d = self.get_train_val_data(seq_val, seq_length, seq_ovelap_rate)

        # Load image features, shape: (num_seqs, seq_length, embedding_size)
        self.train_img = self.load_features(train_d['images'],
                                       train_d['bboxes'],
                                       train_d['ped_ids'],
                                       data_type='train',
                                       load_path=self.get_path(type_save='data',
                                                               data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'], # images    
                                                               model_name='vgg16_'+'none',
                                                               data_subset = 'train')) # shape: (num_seqs, seq_length, 7, 7, 512) using VGG16
        self.val_img = self.load_features(val_d['images'],
                                     val_d['bboxes'],
                                     val_d['ped_ids'],
                                     data_type='val',
                                     load_path=self.get_path(type_save='data',
                                                             data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'],
                                                             model_name='vgg16_'+'none',
                                                             data_subset='val'))        
        # Compute image features: TODO implement Spatial Aggregator

        # Compute sequence features
        # self.train_img = self.temporal_aggregator(train_img) # shape: (num_seqs, embedding_size)
        # self.val_img = self.temporal_aggregator(val_img) # shape: (num_seqs, embedding_size)

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
        import sys
        sys.exit(0)
        self.start_time = time.time()
        for curr_step in range(self.global_step, int(self.train_params['num_steps']+1)):
            self.training_loop()

    def train(self):
        self.build_train_val_data()
        self.build_train_op()

        self.build_sess_and_saver()
        self.init_and_restore()

        self.run_train_loop()
