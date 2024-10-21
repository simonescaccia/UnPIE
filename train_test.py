import sys
import tensorflow as tf

from dataset.pie_preprocessing import PIEPreprocessing
from model.params_loader import ParamsLoader
from model.unpie import UnPIE
from utils.print_utils import print_separator

tf.compat.v1.disable_v2_behavior()

def get_unpie(training_step, train_test):
    is_test = train_test > 0 # 0: train only, 1: train and test, 2: test only

    params_loader = ParamsLoader(training_step)
    pie_preprocessing = PIEPreprocessing(params_loader.get_pie_params())
    data_loaders = pie_preprocessing.get_data_loaders(is_test)
    params = params_loader.get_params(data_loaders, is_test)
    unpie = UnPIE(params)
    return unpie

if __name__ == '__main__':
    train_test = int(sys.argv[1]) 
    training_step = sys.argv[2]
    print_separator('UnPIE started, step: ' + training_step, top_new_line=False)
    
    unpie = get_unpie(training_step, train_test)
    unpie.build_model()

    if train_test < 2:
        unpie.train()
    if train_test > 0:
        unpie.test()
        print_separator('UnPIE ended', bottom_new_line=False)