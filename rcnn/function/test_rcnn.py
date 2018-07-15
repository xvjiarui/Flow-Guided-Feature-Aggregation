# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import argparse
import pprint
import logging
import time
import os
import mxnet as mx
import numpy as np

from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester import Predictor, pred_eval, pred_eval_multiprocess
from utils.load_model import load_param

def get_predictor(sym, sym_instance, cfg, arg_params, aux_params, test_data, ctx):
    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))) ]]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor

def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None, use_philly=False):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    sym = sym_instance.get_symbol(cfg, is_train=False)
    imdb = eval(dataset)(image_set, root_path, dataset_path, cfg.dataset.motion_iou_path, result_path=output_path, use_philly=use_philly)
    roidb = imdb.gt_roidb()

    # get test data iter
    # split roidbs
    gpu_num = len(ctx)
    roidbs = [[] for x in range(gpu_num)]
    roidbs_seg_lens = np.zeros(gpu_num, dtype=np.int)
    for x in roidb:
        gpu_id = np.argmin(roidbs_seg_lens)
        roidbs[gpu_id].append(x)
        roidbs_seg_lens[gpu_id] += x['frame_seg_len']

    # get test data iter
    test_datas = [TestLoader(x, cfg, batch_size=1, shuffle=shuffle, has_rpn=has_rpn) for x in roidbs]

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # create predictor
    predictors = [get_predictor(sym, sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]]) for i in range(gpu_num)]

    # start detection
    #pred_eval(0, key_predictors[0], cur_predictors[0], test_datas[0], imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)
    print("multi process testing with {} gpus".format(gpu_num))
    pred_eval_multiprocess(gpu_num, predictors, test_datas, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)
