import os
import contextlib
import fire
from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # misc
    cfg.use_gpu = True
    cfg.template_cfg = ''
    cfg.cfg_file = ''

    # model
    cfg.model = CN()
    cfg.model.name = 'resnet50'
    cfg.model.pretrained = True  # automatically load pretrained model weights if available
    cfg.model.load_weights = ''  # path to model weights
    cfg.model.resume = ''  # path to checkpoint for resume training
    cfg.model.xv_only = False  # for model vg_interact, whether to test with visual feature only
    cfg.model.xg_only = False  # for model vg_interact, whether to test with gcn feature only
    # for model vg_interact, load a pre-trained visual-only KPPA model.
    cfg.model.kppa_pretrained_path = 'logs/v0.3.2/Jul03_25-53-45_TITAN-RTX/model/model.pth.tar-45'
    cfg.model.gcn_pool_modes = ['G1', 'G2', 'G3']

    # data
    cfg.data = CN()
    cfg.data.type = 'image'
    cfg.data.root = 'reid-data'
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']
    cfg.data.workers = 4  # number of data loading workers
    cfg.data.split_id = 0  # split index
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.transforms = ['random_flip']  # data augmentation
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = 'log'  # path to save log
    cfg.data.load_train_targets = False
    cfg.data.headless = False  # whether to remove keypoints on the head
    cfg.data.rm_glitches = False  # whether to remove bag keypoints during a sequence
    cfg.data.include_dynamics = False  # whther append velocity and acceleration to the position coordinate
    cfg.data.include_spatial_links = True
    cfg.data.include_temporal_links = True
    cfg.data.pose_w_img = False  # whether to return imgs alongside the poses

    # specific datasets
    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = False  # add 500k distractors to the gallery set for market1501
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = False  # use labeled images, if False, use detected images
    cfg.cuhk03.classic_split = False  # use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False  # use cuhk03's metric for evaluation

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomSampler'  # sampler for source train loader
    cfg.sampler.train_sampler_t = 'RandomSampler'  # sampler for target train loader
    cfg.sampler.num_instances = 4  # number of instances per identity for RandomIdentitySampler

    # video reid setting
    cfg.video = CN()
    cfg.video.seq_len = 8  # number of images to sample in a tracklet
    cfg.video.sample_method = 'evenly'  # how to sample images from a tracklet
    cfg.video.pooling_method = 'avg'  # how to pool features over a tracklet

    # train
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers
    cfg.train.open_layers = [
        'classifier'
    ]  # layers for training while keeping others frozen
    cfg.train.staged_lr = False  # set different lr to different layers
    cfg.train.new_layers = ['classifier']  # newly added layers with default lr
    cfg.train.base_lr_mult = 0.1  # learning rate multiplier for base layers
    cfg.train.lr_scheduler = 'single_step'
    cfg.train.stepsize = [20]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier
    cfg.train.print_freq = 20  # print frequency
    cfg.train.seed = 1  # random seed

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.  # dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = 'softmax'
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True  # use label smoothing regularizer
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    cfg.loss.triplet.weight_t = 1.  # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0.  # weight to balance cross entropy loss

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = 'euclidean'  # distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = False  # normalize feature vectors before computing distance
    cfg.test.ranks = [1, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    cfg.test.eval_freq = -1  # evaluation frequency (-1 means to only test after training)
    cfg.test.start_eval = 0  # start to evaluate after a specific epoch
    cfg.test.rerank = False  # use person re-ranking
    cfg.test.visrank = False  # visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank_topk = 10  # top-k ranks to visualize

    return cfg


def parse_from_cli():
    # using fire.Fire silently
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            cfg = fire.Fire(cli_container)
    return cfg


def cli_container(**kwargs):
    cfg = get_default_config()

    # template_cfg
    if 'template_cfg' in kwargs:
        cfg.merge_from_file(kwargs['template_cfg'])
    elif 'cfg_file' in kwargs:
        cfg.merge_from_file(kwargs['cfg_file'])

    # override with cli args
    for k, v in kwargs.items():
        setattribute(cfg, k, v)

    return cfg


def setattribute(obj, name, value):
    '''
    obj: CfgNode object
    '''
    if '.' in name:
        group, name = name.split('.', 1)
        ns = obj[group]
        setattribute(ns, name, value)
        obj[group] = ns
    else:
        if name not in obj:
            print(f' * Warning: keyword \'{name}\' not in cfg, appending...')
        obj[name] = value


def imagedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'load_train_targets': cfg.data.load_train_targets,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        'train_sampler_t': cfg.sampler.train_sampler_t,
        # image
        'cuhk03_labeled': cfg.cuhk03.labeled_images,
        'cuhk03_classic_split': cfg.cuhk03.classic_split,
        'market1501_500k': cfg.market1501.use_500k_distractors,
    }


def videodata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        # video
        'seq_len': cfg.video.seq_len,
        'sample_method': cfg.video.sample_method 
    }


def posedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        # 'norm_mean': cfg.data.norm_mean,
        # 'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        # video
        'seq_len': cfg.video.seq_len,
        'sample_method': cfg.video.sample_method,
        # keypoints
        'headless': cfg.data.headless,
        'rm_glitches': cfg.data.rm_glitches,
        'include_dynamics': cfg.data.include_dynamics,
        'include_spatial_links': cfg.data.include_spatial_links,
        'include_temporal_links': cfg.data.include_temporal_links,
        'return_img': cfg.data.pose_w_img
    }


def optimizer_kwargs(cfg):
    return {
        'optim': cfg.train.optim,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
        'staged_lr': cfg.train.staged_lr,
        'new_layers': cfg.train.new_layers,
        'base_lr_mult': cfg.train.base_lr_mult
    }


def lr_scheduler_kwargs(cfg):
    return {
        'lr_scheduler': cfg.train.lr_scheduler,
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'max_epoch': cfg.train.max_epoch
    }


def engine_run_kwargs(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'max_epoch': cfg.train.max_epoch,
        'start_epoch': cfg.train.start_epoch,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.train.open_layers,
        'start_eval': cfg.test.start_eval,
        'eval_freq': cfg.test.eval_freq,
        'test_only': cfg.test.evaluate,
        'print_freq': cfg.train.print_freq,
        'dist_metric': cfg.test.dist_metric,
        'normalize_feature': cfg.test.normalize_feature,
        'visrank': cfg.test.visrank,
        'visrank_topk': cfg.test.visrank_topk,
        'use_metric_cuhk03': cfg.cuhk03.use_metric_cuhk03,
        'ranks': cfg.test.ranks,
        'rerank': cfg.test.rerank
    }
