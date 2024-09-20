from dataset.pie_data import PIE
import os
import yaml

# (tensorflow 1)
import tensorflow as tf
tf.enable_eager_execution()

with open('config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

pie_path = config_file['PIE_PATH']

os.chdir(pie_path)

imdb = PIE(data_path=pie_path)
sets_to_extract = config_file['SETS_TO_EXTRACT'] or imdb.get_image_set_ids('all')

imdb.extract_images_and_save_features(sets_to_extract)
imdb.organize_features()