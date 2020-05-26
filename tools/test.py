import cv2
import logging
import numpy as np
from collections import defaultdict

import torch

import _init_paths  # pylint: disable=unused-import
from fairmot.datasets.dataset import LoadImages, get_blob
from fairmot.datasets.evaluation import post_processing, write_results
from fairmot.modeling.model_builder import create_model
from fairmot.tracker.multitracker import JDETracker
from fairmot.tracking_utils.log import logger
from fairmot.utils.timer import Timer


def run(model):
    timers = defaultdict(Timer)
    dataloader = LoadImages('data/MOT15/images/train/KITTI-13/img1')
    tracker = JDETracker(0.6, frame_rate=10)
    results = []
    frame_id = 0
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
            results.append((frame_id + 1, online_tlwhs, online_ids))
            frame_id += 1
            timers['asso'].toc()

            timers['all_time'].toc()
            for k, v in timers.items():
                if k != 'all_time':
                    print(' | {}: {:.3f}s'.format(k, v.average_time))
            print('| {}: {:.3f}s'.format('all_time', timers['all_time'].average_time))
    write_results('test/KITTI-13.txt', results, 'mot')


def vis_det(img0, dets):
    dets = dets.astype(np.int)
    for i in range(0, dets.shape[0]):
        bbox = dets[i][0:4]
        cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_tl = bbox[0], bbox[1] - 20
        cv2.putText(img0, '{}'.format(dets[i][4]), txt_tl, font, 1, (0, 255, 255), lineType=cv2.LINE_AA)
    return img0


def test(model):
    timers = defaultdict(Timer)
    video_name = 'test/2P.mp4'
    save_name = 'test/2P_vis.mp4'
    det_th = 0.5

    cap = cv2.VideoCapture(video_name)
    w = int(cap.get(3))
    h = int(cap.get(4) / 2)
    fps = cap.get(5)
    videowriter = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (w, h))
    num_images = int(cap.get(7))

    with torch.no_grad():
        for i in range(num_images):
            timers['all_time'].tic()
            timers['read_img'].tic()
            if (i + 1) % 10 == 0:
                print('======={}/{}======='.format(i + 1, num_images))
            ret, img0 = cap.read()
            img0 = img0[:960, :, :]
            if not ret:
                continue

            timers['read_img'].toc()
            timers['model'].tic()
            img = get_blob(img0)
            output = model(img)
            torch.cuda.synchronize()
            timers['model'].toc()
            timers['post'].tic()
            dets, id_feature = post_processing(output, img, img0)
            timers['post'].toc()

            timers['vis'].tic()
            img_vis = vis_det(img0, dets)
            videowriter.write(img_vis)
            timers['vis'].toc()
            timers['all_time'].toc()

            for k, v in timers.items():
                if k != 'all_time':
                    print(' | {}: {:.3f}s'.format(k, v.average_time))
            print('| {}: {:.3f}s'.format('all_time', timers['all_time'].average_time))


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    model = create_model()
    print(model)

    checkpoint = torch.load('ckpts/all_dla34.pth')['state_dict']
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    # run(model)
    test(model)


