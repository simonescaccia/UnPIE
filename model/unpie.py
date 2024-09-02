import numpy as np

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

        return {'images': images,
                'bboxes': bboxes,
                'ped_ids': ped_ids,
                'output': int_bin}

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
                                                                    data_type='features'+'_'+self.data_opts['crop_type']+'_'+self.data_opts['crop_mode'], # images    
                                                                    model_name='vgg16_'+'none',
                                                                    data_subset = 'train'))

        self.inputs = ''

    def train(self):
        self.build_train()
        self.build_val()

        self.build_sess_and_saver()
        self.init_and_restore()

        self.run_train_loop()
