import sys
import yaml
import os
import data
import tensorflow as tf
import numpy as np

from dataset.pie_preprocessing import PIEPreprocessing
from model import instance_model
from model.unpie import UnPIE
from utils.print_utils import print_separator
from utils.vie_utils import tuple_get_one


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
                for v in tf.compat.v1.trainable_variables()
                if exclude_batch_norm(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all


def loss_func(output):
    return output['loss']


def get_lr_from_boundary_and_ramp_up(
        global_step, boundaries, 
        init_lr, target_lr, ramp_up_epoch,
        num_batches_per_epoch):
    curr_epoch  = tf.math.divide(
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


def get_loss_lr_opt_params_from_arg(args, setting, dataset_len):
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
            'num_batches_per_epoch': dataset_len // args['batch_size'],
            'boundaries': args[setting]['lr_boundaries'],
            'ramp_up_epoch': args['ramp_up_epoch'],
            }

    # optimizer_params: use tfutils optimizer,
    # as mini batch is implemented there
    optimizer_params = {
            'optimizer': tf.compat.v1.train.MomentumOptimizer,
            'momentum': .9,
            }
    return loss_params, learning_rate_params, optimizer_params


def get_save_load_params_from_arg(args, setting):
    # load_params: defining where to load, if needed
    load_dbname = args['db_name']
    load_collname = args['col_name']
    load_exp_id = args[setting]['exp_id']

    # save_params: defining where to save the models
    fre_cache_filter = args['fre_cache_filter'] or args[setting]['fre_filter']
    cache_dir = os.path.join(
            args['cache_dir'], 'models',
            load_dbname, load_collname, load_exp_id)
    save_params = {
            'save_metrics_freq': args['fre_metric'],
            'save_valid_freq': args['fre_valid'],
            'save_filters_freq': args['fre_filter'],
            'cache_filters_freq': fre_cache_filter,
            'cache_dir': cache_dir,
            }

    load_exp = args[setting]['load_exp']
    load_step = args[setting]['load_step']
    load_query = None

    if not args['resume']:
        if load_exp is not None:
            load_dbname, load_collname, load_exp_id = load_exp.split('/')
        if load_step:
            load_query = {'exp_id': load_exp_id,
                          'saved_filters': True,
                          'step': load_step}
            print('Load query', load_query)

    load_params = {
            'dbname': load_dbname,
            'collname': load_collname,
            'exp_id': load_exp_id,
            'query': load_query,
            }
    return save_params, load_params


def get_model_func_params(args, setting, dataset_len):
    model_params = {
        "instance_t": args['instance_t'],
        "instance_k": args['instance_k'],
        "trn_use_mean": args['trn_use_mean'],
        "kmeans_k": args['kmeans_k'],
        "task": args[setting]['task'],
        "instance_data_len": dataset_len
    }
    return model_params


def get_pie_params(config, args):
    pie_path = config['PIE_PATH']
    batch_size = args['batch_size']
    val_batch_size = args['val_batch_size']
    val_num_clips = args['val_num_clips']
    data_opts = {
        'fstride': 1,
        'sample_type': 'all', 
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'data_split_type': 'default',  #  kfold, random, default
        'seq_type': 'intention', #  crossing , intention
        'min_track_size': 0, #  discard tracks that are shorter
        'max_size_observe': args['num_frames'],  # number of observation frames
        'max_size_predict': 5,  # number of prediction frames
        'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
        'balance': True,  # balance the training and testing samples
        'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
        'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
        'encoder_input_type': [],
        'decoder_input_type': ['bbox'],
        'output_type': ['intention_binary']
    }
    pie_params = {
        'data_opts': data_opts,
        'pie_path': pie_path,
        'batch_size': batch_size,
        'val_batch_size': val_batch_size,
        'val_num_clips': val_num_clips
    }
    return pie_params


def rep_loss_func(
        inputs,
        output,
        gpu_offset=0,
        **kwargs
        ):
    data_indx = output['data_indx']
    new_data_memory = output['new_data_memory']
    loss_pure = output['loss']

    memory_bank_list = output['memory_bank']
    all_labels_list = output['all_labels']
    if isinstance(memory_bank_list, tf.Variable):
        memory_bank_list = [memory_bank_list]
        all_labels_list = [all_labels_list]

    devices = ['/gpu:%i' \
               % (idx + gpu_offset) for idx in range(len(memory_bank_list))]
    update_ops = []
    for device, memory_bank, all_labels \
            in zip(devices, memory_bank_list, all_labels_list):
        with tf.device(device):
            mb_update_op = tf.compat.v1.scatter_update(
                    memory_bank, data_indx, new_data_memory)
            update_ops.append(mb_update_op)
            lb_update_op = tf.compat.v1.scatter_update(
                    all_labels, data_indx,
                    inputs['label'])
            update_ops.append(lb_update_op)

    with tf.control_dependencies(update_ops):
        # Force the updates to happen before the next batch.
        loss_pure = tf.identity(loss_pure)

    ret_dict = {'loss_pure': loss_pure}
    for key, value in output.items():
        if key.startswith('loss_'):
            ret_dict[key] = value
    return ret_dict


def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
        # agg_res[k].append(v)
    return agg_res


def valid_perf_func_kNN(
        inputs, output, 
        instance_t,
        k, val_num_clips,
        num_classes):
    curr_dist, all_labels = output
    all_labels = tuple_get_one(all_labels)
    top_dist, top_indices = tf.nn.top_k(curr_dist, k=k)
    top_labels = tf.gather(all_labels, top_indices)
    top_labels_one_hot = tf.one_hot(top_labels, num_classes)
    top_prob = tf.exp(top_dist / instance_t)
    top_labels_one_hot *= tf.expand_dims(top_prob, axis=-1)
    top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
    top_labels_one_hot = tf.reshape(
            top_labels_one_hot,
            [-1, val_num_clips, num_classes])
    top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
    _, curr_pred = tf.nn.top_k(top_labels_one_hot, k=1)
    curr_pred = tf.squeeze(tf.cast(curr_pred, tf.int64), axis=1)
    imagenet_top1 = tf.reduce_mean(
            tf.cast(
                tf.equal(curr_pred, inputs['label']),
                tf.float32))
    return {'top1_{k}NN'.format(k=k): imagenet_top1}


def valid_sup_func(
        inputs, output, 
        val_num_clips):
    num_classes = output.get_shape().as_list()[-1]
    curr_output = tf.nn.softmax(output)
    curr_output = tf.reshape(output, [-1, val_num_clips, num_classes])
    curr_output = tf.reduce_mean(curr_output, axis=1)

    top1_accuracy = tf.nn.in_top_k(curr_output, inputs['label'], k=1)
    top5_accuracy = tf.nn.in_top_k(curr_output, inputs['label'], k=5)
    return {'top1': top1_accuracy, 'top5': top5_accuracy}


def get_valid_loop_from_arg(args, val_data_loader):
    val_counter = [0]
    val_step_num = val_data_loader.dataset.__len__() // args['val_batch_size']
    val_data_enumerator = [enumerate(val_data_loader)]
    def valid_loop(sess, target):
        val_counter[0] += 1
        if val_counter[0] % val_step_num == 0:
            val_data_enumerator.pop()
            val_data_enumerator.append(enumerate(val_data_loader))
        _, (image, label, index) = next(val_data_enumerator[0])
        feed_dict = data.get_feeddict(image, label, index, name_prefix='VAL')
        return sess.run(target, feed_dict=feed_dict)
    return valid_loop, val_step_num


def get_topn_val_data_param_from_arg(args):
    topn_val_data_param = {
            'func': data.get_placeholders,
            'batch_size': args['val_batch_size'],
            'num_frames': args['num_frames'] * args['val_num_clips'],
            'img_emb_size': args['img_emb_size'],
            'multi_frame': True,
            'multi_group': args['val_num_clips'],
            'name_prefix': 'VAL'}
    return topn_val_data_param


def get_params(config, args, setting, train_data_loader, val_data_loader):
    dataset_len = train_data_loader.dataset.__len__()

    loss_params, learning_rate_params, optimizer_params = get_loss_lr_opt_params_from_arg(args, setting, dataset_len)
    save_params, load_params = get_save_load_params_from_arg(args, setting)
    pie_params = get_pie_params(config, args)

    kmeans_k = args['kmeans_k']
    if kmeans_k.isdigit():
        args['kmeans_k'] = [int(kmeans_k)]
    else:
        args['kmeans_k'] = [int(each_k) for each_k in kmeans_k.split(',')]

    # model_params: a function that will build the model
    nn_clusterings = []
    def build_output(inputs, train, **kwargs):
        res = instance_model.build_output(inputs, train, **kwargs)
        if not train:
            return res
        outputs, logged_cfg, clustering = res
        nn_clusterings.append(clustering)
        return outputs, logged_cfg

    model_func_params = get_model_func_params(args, setting, dataset_len)
    model_params = {
        'func': build_output,
        'model_func_params': model_func_params
    }

    first_step = []
    data_enumerator = [enumerate(train_data_loader)]
    def train_loop(sess, train_targets, num_minibatches=1, **params):
        assert num_minibatches==1, "Mini-batch not supported!"

        global_step_vars = [v for v in tf.compat.v1.global_variables() \
                            if 'global_step' in v.name]
        assert len(global_step_vars) == 1
        global_step = sess.run(global_step_vars[0])

        first_flag = len(first_step) == 0
        update_fre = args['clstr_update_fre'] or dataset_len // args['batch_size']
        if (global_step % update_fre == 0 or first_flag) \
                and (nn_clusterings[0] is not None):
            if first_flag:
                first_step.append(1)
            print("Recomputing clusters...")
            new_clust_labels = nn_clusterings[0].recompute_clusters(sess)
            for clustering in nn_clusterings:
                clustering.apply_clusters(sess, new_clust_labels)

        if args['part_vd'] is None:
            data_en_update_fre = dataset_len // args['batch_size']
        else:
            new_length = int(dataset_len * args['part_vd'])
            data_en_update_fre = new_length // args['batch_size']

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
            'batch_size': args['batch_size'], 
            'num_frames': args['num_frames'],
            'img_emb_size': args['img_emb_size'],
            'multi_frame': True,
            'multi_group': None,
            'name_prefix': 'TRAIN'}
    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': args[setting]['train_num_steps'],
            'train_loop': {'func': train_loop},
            }
    
    if not args[setting]['task'] == 'SUP':
        ## Add other loss reports (loss_model, loss_noise)
        train_params['targets'] = {
                'func': rep_loss_func
                }

    # validation_params: control the validation
    topn_val_data_param = get_topn_val_data_param_from_arg(args)

    if not args[setting]['task'] == 'SUP':
        val_targets = {
                'func': valid_perf_func_kNN,
                'k': args['kNN_val'],
                'instance_t': args['instance_t'],
                'val_num_clips': args['val_num_clips'],
                'num_classes': args['kmeans_k']}
    else:
        val_targets = {
                'func': valid_sup_func,
                'val_num_clips': args['val_num_clips']}

    valid_loop, val_step_num = get_valid_loop_from_arg(args, val_data_loader)

    topn_val_param = {
        'data_params': topn_val_data_param,
        'queue_params': None,
        'targets': val_targets,
        'num_steps': val_step_num,
        'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        'online_agg_func': online_agg,
        'valid_loop': {'func': valid_loop}
        }

    validation_params = {
            'topn': topn_val_param,
            }

    params = {
        'pie_path': pie_params['pie_path'],
        'loss_params': loss_params,
        'learning_rate_params': learning_rate_params,
        'optimizer_params': optimizer_params,
        'save_params': save_params,
        'load_params': load_params,
        'model_params': model_params,
        'train_params': train_params,
        'validation_params': validation_params,
        'data_opts': pie_params['data_opts']
    }
    return params


if __name__ == '__main__':
    
    # Setup environment
    print_separator('Setting up the environment', space=False)
    config_file = get_yml_file('config.yml')
    args_file = get_yml_file('args.yml')
    pie_params = get_pie_params(config_file, args_file)
    pie_preprocessing = PIEPreprocessing(pie_params)
    set_environment(config_file)

    # Train and/or test
    if len(sys.argv) == 1:
        train_test = 0 # empty param means train and test
    else:
        train_test = int(sys.argv[1]) # 0: train and test, 1: test only
    
    training_steps = ['vd_3dresnet_IR', 'vd_3dresnet']
    if train_test == 0:
        for step in training_steps:
            train_data_loader, val_data_loader = pie_preprocessing.get_dataloaders()
            params = get_params(config_file, args_file, step, train_data_loader, val_data_loader)
            unpie = UnPIE(params)
            unpie.train()
    if train_test >= 0:
        unpie.test()

    print_separator('UnPIE finished', bottom_new_line=False)