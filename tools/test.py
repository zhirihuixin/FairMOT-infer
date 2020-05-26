import logging
import numpy as np
from collections import defaultdict

import torch

import _init_paths  # pylint: disable=unused-import
from fairmot.datasets.dataset import LoadImages
from fairmot.datasets.evaluation import post_processing
from fairmot.modeling.model_builder import create_model
from fairmot.tracker.multitracker import JDETracker
from fairmot.tracking_utils.log import logger
from fairmot.utils.timer import Timer


if __name__ == '__main__':
    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    timers = defaultdict(Timer)
    model = create_model()
    print(model)

    checkpoint = torch.load('ckpts/all_dla34.pth')['state_dict']
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    dataloader = LoadImages('data/MOT15/images/train/KITTI-13/img1')

    tracker = JDETracker(0.6, frame_rate=10)

    with torch.no_grad():
        for i, (img, img0) in enumerate(dataloader):
            timers['all_time'].tic()
            timers['model'].tic()
            output = model(img)
            torch.cuda.synchronize()
            timers['model'].toc()
            timers['post'].tic()
            dets, id_feature = post_processing(output, img, img0)
            timers['post'].toc()

            timers['asso'].tic()
            online_targets = tracker.update(dets, id_feature)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > 200 and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            timers['asso'].toc()

            timers['all_time'].toc()
            for k, v in timers.items():
                if k!= 'all_time':
                    print(' | {}: {:.3f}s'.format(k, v.average_time))
            print(' | {}: {:.3f}s'.format('all_time', timers['all_time'].average_time))


