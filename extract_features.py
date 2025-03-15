from dataset.pie_data import PIE
import os
import yaml
import sys

from dataset.psi_data import PSI
from model.params_loader import ParamsLoader

with open('settings/config.yml', 'r') as file:
    config_file = yaml.safe_load(file)
with open('settings/args.yml', 'r') as file:
    args_file = yaml.safe_load(file)

if not config_file['IS_GPU']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataset = sys.argv[2]

params = ParamsLoader.get_dataset_params_static(args_file, config_file)
feature_extractor = params['feature_extractor']
feat_input_size = params['feat_input_size']
data_opts = params['data_opts']

if dataset == 'pie':
    pie_path = config_file['PIE_PATH']
    os.chdir(pie_path)
    imdb = PIE(data_path=pie_path, data_opts=data_opts, data_sets='all', feature_extractor=feature_extractor, feat_input_size=feat_input_size)
    sets_to_extract = config_file['PIE_SETS_TO_EXTRACT'] or imdb.get_image_set_ids('all')
    imdb.extract_images_and_save_features(sets_to_extract)
elif dataset == 'psi':
    psi_path = config_file['PSI_PATH']
    os.chdir(psi_path)
    imdb = PSI()
    imdb.extract_images_and_save_features()
else:
    raise ValueError('Invalid dataset name')