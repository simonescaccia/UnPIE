import sys
import tensorflow as tf

from dataset.dataset_preprocessing import DatasetPreprocessing
from model.params_loader import ParamsLoader
from model.unpie import UnPIE
from utils.print_utils import print_separator

def get_unpie(training_step, train_test, dataset, num_kfolds, fold):
    is_test = train_test > 0 # 0: train only, 1: train and test, 2: test only
    is_only_test = train_test == 2

    params_loader = ParamsLoader(training_step, num_kfolds, fold)
    dataset_preprocessing = DatasetPreprocessing(params_loader.get_dataset_params(), dataset)
    datasets = dataset_preprocessing.get_datasets(is_only_test)
    params = params_loader.get_params(datasets, is_test, is_only_test)
    args = params_loader.get_args(dataset)
    unpie = UnPIE(params, args)
    return unpie

if __name__ == '__main__':
    train_test = int(sys.argv[1]) 
    training_step = sys.argv[2]
    dataset = sys.argv[3]
    num_kfolds = None if len(sys.argv) < 5 else int(sys.argv[4])
    if num_kfolds and len(sys.argv) < 6:
        raise ValueError('Please provide the fold number')
    fold = None if len(sys.argv) < 6 else int(sys.argv[5])

    print_separator('UnPIE started, step: ' + training_step, top_new_line=False)
    
    unpie = get_unpie(training_step, train_test, dataset, num_kfolds, fold)

    if train_test < 2:
        unpie.train()
    if train_test > 0:
        unpie.test()
        print_separator('UnPIE ended', bottom_new_line=False)