import ast
import os
import sys
import re
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

print_separator('Plotting training results', bottom_new_line=False)
df_tot_val = pd.DataFrame(columns=['epoch', 'top1'])

val_epoch = 0
training_steps = sys.argv[1:]
for i in tqdm(range(len(training_steps))):
    df_val = pd.DataFrame(columns=['epoch', 'top1'])
    df_train = pd.DataFrame(columns=['step', 'loss'])

    params_loader = ParamsLoader(training_steps[i])
    params = params_loader.get_plot_params()
    
    cache_dir = params['cache_dir'] # Set cache directory
    log_file_path = os.path.join(cache_dir, params['train_log_file'])
    val_log_file_path = os.path.join(cache_dir, params['val_log_file'])
    
    with tqdm(total=os.path.getsize(log_file_path)) as pbar:
        with open(log_file_path, 'r') as file:
            for line in file:
                pbar.update(len(line))
                match = re.search(r'Epoch \d+, Step (\d+) .*-- Loss ([\d\.]+)', line)
                step = int(match.group(1))
                loss = float(match.group(2))
                df_train = pd.concat([
                    df_train if not df_train.empty else None, 
                    pd.DataFrame({'step': [step], 'loss': [loss]})], ignore_index=True)
    
    with tqdm(total=os.path.getsize(val_log_file_path)) as pbar:
        with open(val_log_file_path, 'r') as file:
            for line in file:
                pbar.update(len(line))
                line_strip = line.strip()
                line_split = line_strip.split('validation results: ')[1]
                topn_dict = ast.literal_eval(line_split)
                key = list(topn_dict.keys())[0]
                num_neighbors = int(key.split('_')[1].split('NN')[0])
                top1 = float(list(topn_dict.values())[0])
                df_val = pd.concat([
                    df_val if not df_val.empty else None, 
                    pd.DataFrame({'epoch': [val_epoch], 'top1': [top1]})], ignore_index=True)
                val_epoch += 1
    df_tot_val = pd.concat([
        df_tot_val if not df_tot_val.empty else None, 
        df_val], ignore_index=True)
    
    # save the plot of df_train
    save_plot(df_train, 'step', 'loss', 'Training loss', 'Step', 'Loss', os.path.join(cache_dir, 'train_loss.png'))

    # save the plot of df_val
    save_plot(df_val, 'epoch', 'top1', 'Validation metric', 'Epoch', 'Top1', os.path.join(cache_dir, 'val_metric.png'))

# save the plot of df_val
if len(training_steps) > 0:
    tot_cache_dir = os.path.join(os.path.split(cache_dir)[0], 'tot_metrics')
    os.makedirs(tot_cache_dir, exist_ok=True)
    save_plot(df_tot_val, 'epoch', 'top1', 'Validation metric', 'Epoch', 'Top1', os.path.join(tot_cache_dir, 'tot_val_metric.png'))

