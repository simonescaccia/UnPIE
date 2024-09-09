from dataset.pie_data import PIE
import os
import yaml

with open('config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

pie_path = config_file['PIE_PATH']

os.chdir(pie_path)

sets_to_extract = ['set01', 'set02', 'set03', 'set04', 'set05', 'set06'] or config_file['SETS_TO_EXTRACT']

imdb = PIE(data_path=pie_path)
imdb.extract_images_and_save_features(sets_to_extract)
imdb.organize_features()