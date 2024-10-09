import ast
import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.params_loader import ParamsLoader
from utils.print_utils import print_separator

print_separator('Plotting training results', bottom_new_line=False)
df_tot_val = pd.DataFrame(columns=['step', 'top1'])

val_step = 0
training_steps = sys.argv[1:]
for i in tqdm(range(len(training_steps))):
    df_val = pd.DataFrame(columns=['step', 'top1'])
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
                match = re.search(r'Step (\d+).*loss: ([\d\.]+)', line)
                step = int(match.group(1))
                loss = float(match.group(2))
                df_train = df_train.append({'step': step, 'loss': loss}, ignore_index=True)
    with tqdm(total=os.path.getsize(val_log_file_path)) as pbar:
        with open(val_log_file_path, 'r') as file:
            for line in file:
                pbar.update(len(line))
                line_strip = line.strip()
                line_split = line_strip.split('topn: ')[1]
                topn_dict = ast.literal_eval(line_split)
                key = list(topn_dict.keys())[0]
                frequency = int(key.split('_')[1].split('NN')[0])
                top1 = float(list(topn_dict.values())[0])
                df_val = df_val.append({'step': val_step, 'top1': top1}, ignore_index=True)
                val_step += frequency
    df_tot_val = df_tot_val.append(df_val, ignore_index=True)
    
    # save the plot of df_train
    plt.figure()
    plt.plot(df_train['step'], df_train['loss'])
    plt.title('Training loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(cache_dir, 'train_loss.png'))

    # save the plot of df_val
    plt.figure()
    plt.plot(df_val['step'], df_val['top1'])
    plt.title('Validation metric')
    plt.xlabel('Step')
    plt.ylabel('Top1')
    plt.savefig(os.path.join(cache_dir, 'val_metric.png'))

# save the plot of df_val
tot_cache_dir = os.path.join(os.path.split(os.path.split(cache_dir)[0])[0], 'tot_metrics', os.path.split(cache_dir)[1])
os.makedirs(tot_cache_dir, exist_ok=True)
plt.figure()
plt.plot(df_tot_val['step'], df_tot_val['top1'])
plt.title('Validation metric')
plt.xlabel('Step')
plt.ylabel('Top1')
plt.savefig(os.path.join(tot_cache_dir, 'tot_val_metric.png'))
