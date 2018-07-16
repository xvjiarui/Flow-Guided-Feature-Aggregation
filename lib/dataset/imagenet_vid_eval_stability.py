# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xizhou Zhu
# --------------------------------------------------------

"""
given a imagenet vid imdb, compute mAP
"""

import numpy as np
import os
import cPickle
import cv2
import math as mh
import json
from scipy.optimize import linear_sum_assignment
from collections import Counter
from bbox.bbox_transform import bbox_overlaps
from utils.image import xmlread

import pdb

def get_vid_det_np(video):
    # video[cls][img] = N x 5 array [x1, y1, x2, y2, score]
    cur_video_result = []
    for cls_idx, per_cls_boxes in enumerate(video):
        for id, per_img_boxes in enumerate(per_cls_boxes):
            for box in per_img_boxes:
                det_l = box[0]
                det_t = box[1]
                det_w = box[2] - box[0] + 1
                det_h = box[3] - box[1] + 1
                score = box[4]
                det_array = np.array([id, -1, det_l, det_t, det_w, det_h, score, cls_idx+1, -1, -1, -1])
                cur_video_result.append(det_array)
    return np.vstack(cur_video_result)

def parse_vid_gt(filename, frame_seg_id, class_to_index):
    tree = xmlread(filename)
    cur_frame_result = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        bbox = obj.find('bndbox')
        obj_dict['label'] = class_to_index[obj.find('name').text]
        obj_dict['bbox'] = [float(bbox.find('xmin').text),
                            float(bbox.find('ymin').text),
                            float(bbox.find('xmax').text),
                            float(bbox.find('ymax').text)]
        gt_l = obj_dict['bbox'][0]
        gt_t = obj_dict['bbox'][1]
        gt_w = obj_dict['bbox'][2] - obj_dict['bbox'][0] + 1
        gt_h = obj_dict['bbox'][3] - obj_dict['bbox'][1] + 1
        obj_dict['track_id'] = int(obj.find('trackid').text)
        gt_array = np.array([frame_seg_id, obj_dict['track_id'], gt_l, gt_t, gt_w, gt_h, 1, obj_dict['label'], -1, -1, -1])
        cur_frame_result.append(gt_array)
    return cur_frame_result

def get_stability_err(gt, det):
    gt_x = (gt[0] + gt[2]) / 2.
    gt_y = (gt[1] + gt[3]) / 2.

    det_x = (det[0] + det[2]) / 2.
    det_y = (det[1] + det[3]) / 2.

    gt_w = gt[2] - gt[0] + 1
    gt_h = gt[3] - gt[1] + 1
    det_w = det[2] - det[0] + 1
    det_h = det[3] - det[1] + 1

    return (det_x - gt_x) * 1. / gt_w, (det_y - gt_y) * 1. / gt_h, \
            (1. * det_w / det_h) / (1. * gt_w / gt_h)-1,\
            np.sqrt(1. * det_w * det_h) / np.sqrt(1. * gt_w * gt_h)-1

def count_continue_seq(seq):
    if len(seq) == 1: return 0
    n = len(seq) - 1
    frag = [seq[i] != seq[i - 1] for i in xrange(1, len(seq))]
    return np.sum(frag) * 1. / n

def traj_err(gt_traj_frame, det_traj_frame):

    err_F = []
    err_C = []
    err_R = []

    for traj_id in gt_traj_frame:

        if not traj_id in det_traj_frame.keys(): continue

        traj_err = det_traj_frame[traj_id]

        err_x_list = [x[1] for x in traj_err]
        err_y_list = [x[2] for x in traj_err]
        err_r_list = [x[3] for x in traj_err]
        err_s_list = [x[4] for x in traj_err]

        err_C.append(np.std(err_x_list) + np.std(err_y_list))
        err_R.append(np.std(err_r_list) + np.std(err_s_list))

        det_frame_seg_id = set(sorted([x[0] for x in traj_err]))
        gt_frame_seg_id = sorted(gt_traj_frame[traj_id])
        match_seq = []
        for gt in gt_frame_seg_id:
            if gt in det_frame_seg_id: 
                match_seq.append(1)
            else: 
                match_seq.append(0)

        frag = count_continue_seq(match_seq)

        err_F.append(frag)

    return np.mean(err_F), np.mean(err_C), np.mean(err_R)

def assign_eval(gt_traj, det_traj, vid_path, score_thresh, overlap_thresh=0.5, class_agnostic=True):

    # dict {traj_id: [frame_seg_id, err_x, err_y, err_r, err_s]}
    det_traj_frame = {}

    # dict {frame_seg_id: [traj_id, x_min, y_min, x_max, y_max]}
    vid_traj = {}

    def assign_bbox(gt_bbox, det_bbox, frame_seg_id, traj_id):
        if len(gt_bbox) == 0 or len(det_bbox) ==0 :
            return
        overlap_mat = bbox_overlaps(gt_bbox, det_bbox)
        matched_list = linear_sum_assignment(-overlap_mat)
        for matched_gt, matched_det in zip(*matched_list):
            if overlap_mat[matched_gt, matched_det] < overlap_thresh:
                continue
            matched_traj = traj_id[matched_gt]

            matched_gt_bbox = gt_bbox[matched_gt, :]
            matched_det_bbox = det_bbox[matched_det, :]

            err_x, err_y, err_r, err_s  = get_stability_err(matched_gt_bbox, matched_det_bbox)
            
            det_traj_frame.setdefault(matched_traj, []).append([frame_seg_id, err_x, err_y, err_r, err_s])

            vid_traj.setdefault(frame_seg_id, []).append([matched_traj] + list(matched_det_bbox[:4]))

    for frame_seg_id in sorted(det_traj.keys()):
        if not frame_seg_id in gt_traj.keys(): continue
        cur_gt = gt_traj[frame_seg_id]
        cur_gt_bbox = cur_gt['bbox'].astype(float)
        cur_traj_id = cur_gt['traj_id']
        cur_det_bbox = det_traj[frame_seg_id]['bbox'].astype(float)
        cur_det_bbox = cur_det_bbox[cur_det_bbox[:, 4] >=score_thresh]

        if class_agnostic:
            assign_bbox(cur_gt_bbox, cur_det_bbox, frame_seg_id, cur_traj_id)
        else:
            cur_gt_classes = cur_gt_bbox[:, 5].astype(int)
            for cls in list(set(cur_gt_classes)):
                per_cls_det_bbox = cur_det_bbox[cur_det_bbox[:, 5].astype(int) == cls]
                per_cls_gt_bbox = cur_gt_bbox[cur_gt_bbox[:, 5].astype(int) == cls]
                assign_bbox(per_cls_gt_bbox, per_cls_det_bbox, frame_seg_id, cur_traj_id)

    # make traj_id as key
    # dict {traj_id: frame_seg_id}
    gt_traj_frame = {}
    for frame_seg_id, cur_gt in gt_traj.iteritems():
        cur_traj_id = cur_gt['traj_id']
        for traj_id in cur_traj_id:
            gt_traj_frame.setdefault(traj_id, []).append(frame_seg_id)

    F_err, var_c, var_r = traj_err(gt_traj_frame, det_traj_frame)

    vid_out = json.dumps(vid_traj)
    with open(os.path.join(vid_path, 'traj.json'), 'w') as f:
        f.write(vid_out)

    return F_err, var_c, var_r
