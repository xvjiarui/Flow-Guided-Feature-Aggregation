# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuqing Zhu, Shuhao Fu, Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import _init_paths

import cv2
import time
import argparse
import logging
import pprint
import os
import sys
from config.config import config, update_config, update_philly_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--usePhilly', help='experiment is on the philly', action='store_true')

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)
    config.USE_PHILLY = args.usePhilly

    # modfication for philly
    if config.USE_PHILLY:
        # not working actually
        config.gpus = '0,1,2,3,4,5,6,7'
        parser.add_argument('--dataDir', help='input directory for Philly jobs', required=True, type=str)
        parser.add_argument('--modelDir', help='output directory for Philly jobs', required=True, type=str)
        args, rest = parser.parse_known_args()
        update_philly_config(args.modelDir, args.dataDir)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import shutil
import numpy as np
import mxnet as mx

from symbols import *
from core import callback, metric
from core.loader import AnchorLoader
from core.module import MutableModule
from utils.create_logger import create_logger
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler

def train_net(args, ctx, pretrained_dir, pretrained_resnet, pretrained_flow, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    # load symbol
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_train_symbol(config)
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    roidbs = [load_gt_roidb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
                            flip=config.TRAIN.FLIP, use_philly=config.USE_PHILLY)
              for image_set in image_sets]
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, config)
    # load training data
    train_data = AnchorLoader(feat_sym, roidb, config, batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE, ctx=ctx,
                              feat_stride=config.network.RPN_FEAT_STRIDE, anchor_scales=config.network.ANCHOR_SCALES,
                              anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING,
                              normalize_target=config.network.NORMALIZE_RPN, bbox_mean=config.network.ANCHOR_MEANS,
                              bbox_std=config.network.ANCHOR_STDS)

    # infer max shape
    # times 2 here for double images
    max_data_shape = [('data', (2*config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (config.TRAIN.BATCH_IMAGES, 100, 5)))
    max_data_shape.append(('ref_gt_boxes', (config.TRAIN.BATCH_IMAGES, 100, 5)))

    print 'providing maximum shape', max_data_shape, max_label_shape

    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data_single]
    label_names = [k[0] for k in train_data.provide_label_single]

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in range(batch_size)],
                        max_label_shapes=[max_label_shape for _ in range(batch_size)], fixed_param_prefix=fixed_param_prefix)

    # load and initialize params
    params_loaded = False
    if config.TRAIN.RESUME:
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
        # mod._preload_opt_states = '{}-{:04d}.states'.format(prefix, begin_epoch)
        print('continue training from ', begin_epoch)
        logger.info('continue training from ', begin_epoch)
        params_loaded = True
        sym_instance.init_weight(config, arg_params, aux_params)
        print('init nms')
        logger.info('init nms')
    elif config.TRAIN.AUTO_RESUME:
        for cur_epoch in range(end_epoch-1, begin_epoch, -1):
            params_filename = '{}-{:04d}.params'.format(prefix, cur_epoch)
            states_filename = '{}-{:04d}.states'.format(prefix, cur_epoch)
            if os.path.exists(params_filename) and os.path.exists(states_filename):
                begin_epoch = cur_epoch
                arg_params, aux_params = load_param(prefix, cur_epoch, convert=True)
                mod._preload_opt_states = states_filename
                print('auto continue training from {}, {}'.format(params_filename, states_filename))
                logger.info('auto continue training from {}, {}'.format(params_filename, states_filename))
                params_loaded = True
                break
    if not params_loaded:
        arg_params, aux_params = load_param(os.path.join(pretrained_dir, pretrained_resnet), epoch, convert=True)
        arg_params_flow, aux_params_flow = load_param(os.path.join(pretrained_dir, pretrained_flow), epoch, convert=True)
        arg_params.update(arg_params_flow)
        aux_params.update(aux_params_flow)
        sym_instance.init_weight(config, arg_params, aux_params)

    # if config.TRAIN.RESUME:
    #     print('continue training from ', begin_epoch)
    #     arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    # else:
    #     arg_params, aux_params = load_param(pretrained, epoch, convert=True)
    #     arg_params_flow, aux_params_flow = load_param(pretrained_flow, epoch, convert=True)
    #     arg_params.update(arg_params_flow)
    #     aux_params.update(aux_params_flow)

    #     sym_instance.init_weight(config, arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # decide training params
    # metric
    eval_metric = metric.RCNNAccMetric(config)
    cls_metric = metric.RCNNLogLossMetric(config)
    bbox_metric = metric.RCNNL1LossMetric(config)
    eval_metrics = mx.metric.CompositeEvalMetric()

    for child_metric in [eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    if config.TRAIN.JOINT_TRAINING or (not config.TRAIN.LEARN_NMS):
        rpn_eval_metric = metric.RPNAccMetric()
        rpn_cls_metric = metric.RPNLogLossMetric()
        rpn_bbox_metric = metric.RPNL1LossMetric()
        for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric]:
            eval_metrics.add(child_metric)
    if config.TRAIN.LEARN_NMS:
        eval_metrics.add(metric.NMSLossMetric(config, 'pos'))
        eval_metrics.add(metric.NMSLossMetric(config, 'neg'))
        eval_metrics.add(metric.NMSAccMetric(config))

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True), callback.do_checkpoint(prefix, means, stds)]
    # decide learning rate
    base_lr = lr
    lr_factor = config.TRAIN.lr_factor
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)
    # optimizer
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}

    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def main():
    print("Pulled from current commit:")
    os.system('git rev-parse HEAD')
    print('Called with argument:', args)
    if args.usePhilly:
        config.gpus ='0,1,2,3,4,5,6,7'
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]

    train_net(args, ctx, config.network.pretrained_dir, config.network.pretrained_resnet, config.network.pretrained_flow, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)

if __name__ == '__main__':
    main()
