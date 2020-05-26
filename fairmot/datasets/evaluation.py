import numpy as np

import torch
import torch.nn.functional as F

from fairmot.modeling.decode import mot_decode
from fairmot.modeling.utils import _tranpose_and_gather_feat
from fairmot.utils.post_process import ctdet_post_process


def post_processing(blob, im_blob, img0):
    width = img0.shape[1]
    height = img0.shape[0]
    inp_height = im_blob.shape[2]
    inp_width = im_blob.shape[3]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
    meta = {'c': c, 's': s,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4}

    hm = blob['hm'].sigmoid_()
    wh = blob['wh']
    id_feature = blob['id']
    id_feature = F.normalize(id_feature, dim=1)
    reg = blob['reg']

    dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=F, K=128)
    id_feature = _tranpose_and_gather_feat(id_feature, inds)
    id_feature = id_feature.squeeze(0)
    id_feature = id_feature.cpu().numpy()

    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], 1)
    for j in range(1, 1 + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    dets = dets[0]

    dets = merge_outputs([dets])[1]

    remain_inds = dets[:, 4] > 0.6
    dets = dets[remain_inds]
    id_feature = id_feature[remain_inds]

    return dets, id_feature


def merge_outputs(detections):
    results = {}
    for j in range(1, 1 + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, 1 + 1)])
    if len(scores) > 128:
        kth = len(scores) - 128
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, 1 + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results