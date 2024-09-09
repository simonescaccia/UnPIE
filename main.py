import sys
import yaml
import os

from dataset.pie_data import PIE
from model.unpie import UnPIE

def get_config_file():
    with open('config.yml', 'r') as file:
        config_file = yaml.safe_load(file)
    return config_file

def set_environment(config_file):
    if not config_file['IS_GPU']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def get_params(config_file):
    params = {
        'pie_path': config_file['PIE_PATH']
    }
    return params

if __name__ == '__main__':

    # Setup environment
    config_file = get_config_file()
    set_environment(config_file)

    # Laod parameters
    params = get_params(config_file)

    # Train and/or test
    if len(sys.argv) == 1:
        train_test = 0 # empty param means train and test
    else:
        train_test = int(sys.argv[1]) # 0: train and test, 1: test only
    
    unpie = UnPIE(params)

    if train_test == 0:
        unpie.train()
    if train_test >= 0:
        unpie.test()