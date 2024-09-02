import sys
import yaml
import os

from dataset.pie_data import PIE
from model.unpie import UnPIE

if __name__ == '__main__':

    # Setup environment
    with open('config.yml', 'r') as file:
        config_file = yaml.safe_load(file)

    if not config_file['IS_GPU']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Laod parameters

    # Train and/or test
    train_test = int(sys.argv[1]) # 0: train and test, 1: test only
    
    unpie = UnPIE(params)

    if train_test == 0:
        unpie.train()
    if train_test >= 0:
        unpie.test()