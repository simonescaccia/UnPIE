import ast
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.params_loader import ParamsLoader
from utils.print_utils import print_separator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_cluster(save_path):
    algorithm = 'PCA_TSNE'
    # load the true labels
    file_name = os.path.join(save_path, 'true_labels.npy')
    y = np.load(file_name)

    # load the memory bank for each epoch
    memory_bank_path = os.path.join(save_path, 'memory_bank')
    if not os.path.exists(memory_bank_path):
        return
    # read all files in the directory
    files = os.listdir(memory_bank_path)
    for file in tqdm(files):
        # get the epoch number from the file name
        epoch = file.split('_')[1]
        # read the memory bank
        memory_bank = np.load(os.path.join(memory_bank_path, file))
        # reduce the data dimention using PCA and 
        pca = PCA(n_components=50)
        tsne = TSNE(n_components=2)
        data_pca = pca.fit_transform(memory_bank)
        data_tsne = tsne.fit_transform(data_pca)
        # plot clusters
        plot_cluster_epoch(save_path, data_tsne, epoch, y, algorithm)


def plot_cluster_epoch(save_path, data_2d, epoch, y, algorithm):
    alg_path = os.path.join(save_path, algorithm)

    # Create a figure for the true labels plot
    plt.figure(figsize=(7, 7))
    unique_labels = np.unique(y)
    for label in unique_labels:
        label_points = data_2d[y == label]
        plt.scatter(label_points[:, 0], label_points[:, 1], label=f'Label {label}')
    plt.title('{}: True Labels Visualized in 2D'.format(algorithm))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()

    # Save the figure
    os.system('mkdir -p %s' % alg_path)
    file_name = os.path.join(alg_path, f'epoch_{epoch}_{algorithm}.png')
    plt.savefig(file_name)
    plt.close()  # Close the figure to avoid interactive display

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
    
    # plot clusters
    plot_save_path = os.path.join(cache_dir, params['plot_dir'])
    plot_cluster(plot_save_path)

