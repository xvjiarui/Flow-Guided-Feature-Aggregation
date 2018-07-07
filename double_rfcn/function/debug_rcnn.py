# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import argparse
import pprint
import logging
import time
import os
import numpy as np
import mxnet as mx

from symbols import *
from dataset import *
from core.loader import VisTestLoader
from core.debugger import Predictor, pred_double_eval
from utils.load_model import load_param


def get_debug_predictor(sym, sym_instance, cfg, arg_params, aux_params, test_data, ctx):
    # infer shape
    # data_shape_dict = dict(test_data.provide_data)
    # sym_instance.infer_shape(data_shape_dict)
    # sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    # label_names = [k[0] for k in test_data.provide_label_single]
    label_names = None
    max_data_shape = [[('data', (2, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),]]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor

def debug_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, show_gt, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    sym = sym_instance.get_vis_test_symbol(cfg)
    imdb = eval(dataset)(image_set, root_path, dataset_path, cfg.dataset.motion_iou_path, result_path=output_path)
    roidb = imdb.gt_roidb()

    # get test data iter
    test_data = VisTestLoader(roidb, cfg, batch_size=1, shuffle=shuffle, has_rpn=has_rpn)
    # load model

    data_shape_dict = dict(test_data.provide_data_single)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    arg_params, aux_params = load_param(prefix, epoch, process=True)
    # sym_instance.init_weight(cfg, arg_params, aux_params)

    # create predictor
    predictor = get_debug_predictor(sym, sym_instance, cfg, arg_params, aux_params, test_data, ctx)

    # start detection
    #pred_eval(0, key_predictors[0], cur_predictors[0], test_datas[0], imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)
    pred_double_eval(predictor, test_data, imdb, cfg, vis=vis, show_gt=show_gt, ignore_cache=ignore_cache, thresh=thresh, logger=logger)