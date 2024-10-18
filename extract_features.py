from dataset.pie_data import PIE
import os
import yaml
import tensorflow as tf

with open('settings/config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

pie_path = config_file['PIE_PATH']

os.chdir(pie_path)

imdb = PIE(data_path=pie_path)
sets_to_extract = config_file['SETS_TO_EXTRACT'] or imdb.get_image_set_ids('all')

imdb.extract_images_and_save_features(sets_to_extract)