import sys
import yaml
import os

import tensorflow as tf

import data

from model import instance_model
from model.unpie import UnPIE

def get_yml_file(name):
    with open(name, 'r') as file:
        yml_file = yaml.safe_load(file)
    return yml_file

def set_environment(config_file):
    if not config_file['IS_GPU']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def reg_loss(loss, weight_decay):
    # Add weight decay to the loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name
    l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in tf.trainable_variables()
                if exclude_batch_norm(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all

def loss_func(output):
    return output['loss']

def get_lr_from_boundary_and_ramp_up(
        global_step, boundaries, 
        init_lr, target_lr, ramp_up_epoch,
        num_batches_per_epoch):
    curr_epoch  = tf.div(
            tf.cast(global_step, tf.float32), 
            tf.cast(num_batches_per_epoch, tf.float32))
    curr_phase = (tf.minimum(curr_epoch/float(ramp_up_epoch), 1))
    curr_lr = init_lr + (target_lr-init_lr) * curr_phase

    if boundaries is not None:
        boundaries = boundaries.split(',')
        boundaries = [int(each_boundary) for each_boundary in boundaries]

        all_lrs = [
                curr_lr * (0.1 ** drop_level) \
                for drop_level in range(len(boundaries) + 1)]

        curr_lr = tf.train.piecewise_constant(
                x=global_step,
                boundaries=boundaries, values=all_lrs)
    return curr_lr

def get_loss_lr_opt_params_from_arg(args, setting):
    # loss_params: parameters to build the loss
    loss_params = {
        'pred_targets': [],
        'agg_func': reg_loss,
        'agg_func_kwargs': {'weight_decay': args['weight_decay']},
        'loss_func': loss_func,
    }

    # learning_rate_params: build the learning rate
    # For now, just stay the same
    learning_rate_params = {
            'func': get_lr_from_boundary_and_ramp_up,
            'init_lr': args['init_lr'],
            'target_lr': args['target_lr'] or args['init_lr'],
            'boundaries': args[setting]['lr_boundaries'],
            'ramp_up_epoch': args['ramp_up_epoch'],
            }

    # optimizer_params: use tfutils optimizer,
    # as mini batch is implemented there
    optimizer_params = {
            'optimizer': tf.train.MomentumOptimizer,
            'momentum': .9,
            }
    return loss_params, learning_rate_params, optimizer_params

def get_save_load_params_from_arg(args, setting):
    # load_params: defining where to load, if needed
    load_dbname = args[setting]['db_name']
    load_collname = args[setting]['col_name']
    load_exp_id = args[setting]['exp_id']

    # save_params: defining where to save the models
    fre_cache_filter = args[setting]['fre_cache_filter'] or args[setting]['fre_filter']
    cache_dir = os.path.join(
            args['cache_dir'], 'models',
            load_dbname, load_collname, load_exp_id)
    save_params = {
            'save_metrics_freq': args['fre_metric'],
            'save_valid_freq': args[setting]['fre_valid'],
            'save_filters_freq': args[setting]['fre_filter'],
            'cache_filters_freq': fre_cache_filter,
            'cache_dir': cache_dir,
            }

    load_params = {
            'dbname': load_dbname,
            'collname': load_collname,
            'exp_id': load_exp_id
            }
    return save_params, load_params

def get_model_func_params(args, setting):
    model_params = {
        "instance_t": args['instance_t'],
        "instance_k": args['instance_k'],
        "trn_use_mean": args['trn_use_mean'],
        "kmeans_k": args['kmeans_k'],
        "task": args[setting]['task'],
        "num_classes": args['num_classes'],
    }
    return model_params

def get_params(config, args, setting):
    loss_params, learning_rate_params, optimizer_params = get_loss_lr_opt_params_from_arg(args, setting)
    save_params, load_params = get_save_load_params_from_arg(args, setting)

    train_data_loader = get_train_pt_loader_from_arg(args)
    dataset_len = train_data_loader.dataset.__len__()

    # model_params: a function that will build the model
    model_func_params = get_model_func_params(args, setting)
    model_func_params["instance_data_len"] = dataset_len
    nn_clusterings = []
    first_step = []
    def build_output(inputs, outputs, train, **kwargs):
        res = instance_model.build_output(inputs, outputs, train, **model_func_params)
        if not train:
            return res
        outputs, logged_cfg, clustering = res
        nn_clusterings.append(clustering)
        return outputs, logged_cfg

    model_params = {'func': build_output}

    data_enumerator = [enumerate(train_data_loader)]
    def train_loop(sess, train_targets, num_minibatches=1, **params):
        assert num_minibatches==1, "Mini-batch not supported!"

        global_step_vars = [v for v in tf.global_variables() \
                            if 'global_step' in v.name]
        assert len(global_step_vars) == 1
        global_step = sess.run(global_step_vars[0])

        first_flag = len(first_step) == 0
        update_fre = args.clstr_update_fre or dataset_len // args.batch_size
        if (global_step % update_fre == 0 or first_flag) \
                and (nn_clusterings[0] is not None):
            if first_flag:
                first_step.append(1)
            print("Recomputing clusters...")
            new_clust_labels = nn_clusterings[0].recompute_clusters(sess)
            for clustering in nn_clusterings:
                clustering.apply_clusters(sess, new_clust_labels)

        if args.part_vd is None:
            data_en_update_fre = dataset_len // args.batch_size
        else:
            new_length = int(dataset_len * args.part_vd)
            data_en_update_fre = new_length // args.batch_size

        # TODO: make this smart
        if global_step % data_en_update_fre == 0:
            data_enumerator.pop()
            data_enumerator.append(enumerate(train_data_loader))
        _, (image, label, index) = next(data_enumerator[0])
        feed_dict = data.get_feeddict(image, label, index)
        sess_res = sess.run(train_targets, feed_dict=feed_dict)
        return sess_res

    # train_params: parameters about training data
    train_data_param = {
            'func': data.get_placeholders,
            'batch_size': args[setting]['batch_size'], 
            'num_frames': 15,
            'crop_size': 112,
            'multi_frame': True}
    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': args[setting]['train_num_steps'],
            'train_loop': {'func': train_loop},
            }

    params = {
        'pie_path': config['PIE_PATH'],
        'loss_params': loss_params,
        'learning_rate_params': learning_rate_params,
        'optimizer_params': optimizer_params,
        'save_params': save_params,
        'load_params': load_params,
        'model_params': model_params,
        'train_params': train_params
    }
    return params

if __name__ == '__main__':

    # Setup environment
    config_file = get_yml_file('config.yml')
    args_file = get_yml_file('args.yml')
    set_environment(config_file)

    # Train and/or test
    if len(sys.argv) == 1:
        train_test = 0 # empty param means train and test
    else:
        train_test = int(sys.argv[1]) # 0: train and test, 1: test only
    
    if train_test == 0:
        params = get_params(config_file, args_file, 'vd_3dresnet_IR')
        unpie = UnPIE(params)
        unpie.train()
        params = get_params(config_file, args_file, 'vd_3dresnet')
        unpie = UnPIE(params)
        unpie.train()
    if train_test >= 0:
        unpie.test()