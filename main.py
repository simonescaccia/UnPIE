import sys
import yaml

from dataset.pie_data import PIE

with open('config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

def train(self, pie: PIE, data_opts: dict):

    beh_seq_train = pie.generate_data_trajectory_sequence('train', **data_opts)
    beh_seq_train = pie.balance_samples_count(beh_seq_train, label_type='intention_binary')

    beh_seq_val = pie.generate_data_trajectory_sequence('val', **data_opts)
    beh_seq_val = pie.balance_samples_count(beh_seq_val, label_type='intention_binary')

if __name__ == '__main__':

    train_test = int(sys.argv[1]) # 1: train and test, 2: test only

    data_opts = {'fstride': 1,
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

    pie_path = config_file['PIE_PATH']
    pie = PIE(data_path=pie_path)
    
    if train_test < 2:
        train(pie=pie, data_opts=data_opts)
    if train_test == 2:
        pass