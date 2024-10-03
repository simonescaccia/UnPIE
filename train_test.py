import sys

from dataset.pie_preprocessing import PIEPreprocessing
from model.params_loader import ParamsLoader
from model.unpie import UnPIE
from utils.print_utils import print_separator


if __name__ == '__main__':
    train_test = int(sys.argv[1]) # 0: train only, 1: train and test, 2: test only
    training_step = sys.argv[2]
    print_separator('UnPIE started, step: ' + training_step, top_new_line=False)

    params_loader = ParamsLoader(training_step)
    pie_preprocessing = PIEPreprocessing(params_loader.get_pie_params())
    train_data_loader, val_data_loader = pie_preprocessing.get_dataloaders()
    params = params_loader.get_params(train_data_loader, val_data_loader)
    unpie = UnPIE(params)

    if train_test < 2:
        unpie.train()
    if train_test > 0:
        unpie.test()
        print_separator('UnPIE ended', bottom_new_line=False)