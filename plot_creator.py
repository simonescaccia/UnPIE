import ast
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.params_loader import ParamsLoader
from utils.print_utils import print_separator

def save_plot(df, x_col, y_col, title, xlabel, ylabel, save_path):
    """
    Plot and save a line plot from a DataFrame.
    :param df: DataFrame with the data to plot.
    :param x_col: Column for the x-axis.
    :param y_col: Column for the y-axis.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param save_path: Path to save the plot.
    """
    plt.figure()
    plt.plot(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()

def get_df(file_path):
    df = pd.DataFrame()
    with tqdm(total=os.path.getsize(file_path)) as pbar:
        with open(file_path, 'r') as file:
            for line in file:
                pbar.update(len(line))
                line_strip = line.strip()
                line_dict = ast.literal_eval(line_strip)
                df = pd.concat([
                    df if not df.empty else None, 
                    pd.DataFrame([line_dict])], ignore_index=True)   
    return df


print_separator('Plotting training results', bottom_new_line=False)
df_tot_val = pd.DataFrame(columns=['epoch', 'top1'])

training_steps = sys.argv[1:]
for i in tqdm(range(len(training_steps))):
    params_loader = ParamsLoader(training_steps[i])
    params = params_loader.get_plot_params()
    
    cache_dir = params['cache_dir'] # Set cache directory
    log_file_path = os.path.join(cache_dir, params['train_log_file'])
    val_log_file_path = os.path.join(cache_dir, params['val_log_file'])
    
    train_df = get_df(log_file_path)
    val_df = get_df(val_log_file_path)
    
    # save train_df plot
    save_plot(train_df, 'Step', 'Loss', 'Training loss', 'Step', 'Loss', os.path.join(cache_dir, 'train_loss.png'))
    save_plot(train_df, 'Step', 'Learning rate', 'Training learning rate', 'Step', 'Learning rate', os.path.join(cache_dir, 'train_lr.png'))

    # save val_df plot
    df_keys = list(val_df.keys())
    df_keys.remove('Epoch')
    for key in df_keys:
        save_plot(val_df, 'Epoch', key, 'Validation {}'.format(key), 'Epoch', key, os.path.join(cache_dir, 'val_{}.png'.format(key.lower())))

