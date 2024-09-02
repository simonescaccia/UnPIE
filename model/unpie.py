from dataset.pie_data import PIE


class UnPIE(object):
    def __init__(self, params):
        self.params = params
        self.pie = PIE(data_path=params['pie_path'])
        self.data_opts = {
            'fstride': 1,
            'sample_type': 'all', 
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',  #  kfold, random, default
            'seq_type': 'intention', #  crossing , intention
            'min_track_size': 0, #  discard tracks that are shorter
            'max_size_observe': 15,  # number of observation frames
            'max_size_predict': 5,  # number of prediction frames
            'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
            'balance': True,  # balance the training and testing samples
            'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
            'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
            'encoder_input_type': [],
            'decoder_input_type': ['bbox'],
            'output_type': ['intention_binary']
        }
        self.data_type = {
            'encoder_input_type': self.data_opts['encoder_input_type'],
            'decoder_input_type': self.data_opts['decoder_input_type'],
            'output_type': self.data_opts['output_type']
        }

    def build_train(self):
        '''
        Build the inputs for the embedding computation
        '''
        seq_train = self.pie.generate_data_trajectory_sequence('train', **self.data_opts)
        seq_train = self.pie.balance_samples_count(seq_train, label_type='intention_binary')

        seq_val = self.pie.generate_data_trajectory_sequence('val', **self.data_opts)
        seq_val = self.pie.balance_samples_count(seq_val, label_type='intention_binary')
        
        seq_length = self.data_opts['max_size_observe']
        seq_ovelap_rate = self.data_opts['seq_overlap_rate']
        train_d = self.get_train_val_data(seq_train, self.data_type, seq_length, seq_ovelap_rate)
        val_d = self.get_train_val_data(seq_val, self.data_type, seq_length, seq_ovelap_rate)

        train_img = self.load_images_and_process(train_d['images'],
                                            train_d['bboxes'],
                                            train_d['ped_ids'],
                                            data_type='train',
                                            save_path=self.get_path(type_save='data',
                                                                    data_type='features'+'_'+data_opts['crop_type']+'_'+data_opts['crop_mode'], # images    
                                                                    model_name='vgg16_'+'none',
                                                                    data_subset = 'train'))

        self.inputs = ''

    def train(self):
        self.build_train()
        self.build_val()

        self.build_sess_and_saver()
        self.init_and_restore()

        self.run_train_loop()
