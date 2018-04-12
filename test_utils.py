from utils import get_filtered_lidar, project_velo2rgb, draw_rgb_projections
from config import config as cfg
from data.kitti import KittiDataset
import torch.utils.data as data
from nms.pth_nms import pth_nms
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn
import cv2
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True

def delta_to_boxes3d(deltas, anchors):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)

    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    N = deltas.shape[0]
    deltas = deltas.view(N, -1, 7)
    anchors = torch.FloatTensor(anchors)
    boxes3d = torch.zeros_like(deltas)

    if deltas.is_cuda:
        anchors = anchors.cuda()
        boxes3d = boxes3d.cuda()

    anchors_reshaped = anchors.view(-1, 7)

    anchors_d = torch.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)

    anchors_d = anchors_d.repeat(N, 2, 1).transpose(1,2)
    anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

    boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[...,[3]]) + anchors_reshaped[..., [2]]

    boxes3d[..., [3, 4, 5]] = torch.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]

    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    return boxes3d

def detection_collate(batch):
    lidars = []
    images = []
    calibs = []

    targets = []
    pos_equal_ones=[]
    ids = []
    for i, sample in enumerate(batch):
        lidars.append(sample[0])
        images.append(sample[1])
        calibs.append(sample[2])
        targets.append(sample[3])
        pos_equal_ones.append(sample[4])
        ids.append(sample[5])
    return lidars,images,calibs,\
           torch.cuda.FloatTensor(np.array(targets)), \
           torch.cuda.FloatTensor(np.array(pos_equal_ones)),\
           ids


def box3d_center_to_corner_batch(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = torch.zeros((N, 8, 3))
    if boxes_center.is_cuda:
        ret = ret.cuda()

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = torch.FloatTensor([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])
        if boxes_center.is_cuda:
            trackletBox = trackletBox.cuda()
        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = torch.FloatTensor([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        if boxes_center.is_cuda:
            rotMat = rotMat.cuda()

        cornerPosInVelo = torch.mm(rotMat, trackletBox) + translation.repeat(8, 1).t()
        box3d = cornerPosInVelo.transpose(0,1)
        ret[i] = box3d

    return ret

def box3d_corner_to_top_batch(boxes3d, use_min_rect=True):
    # [N,8,3] -> [N,4,2] -> [N,8]
    box3d_top=[]

    num =len(boxes3d)
    for n in range(num):
        b   = boxes3d[n]
        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        box3d_top.append([x0,y0,x1,y1,x2,y2,x3,y3])

    if use_min_rect:
        box8pts = torch.FloatTensor(np.array(box3d_top))
        if boxes3d.is_cuda:
            box8pts = box8pts.cuda()
        min_rects = torch.zeros((box8pts.shape[0], 4))
        if boxes3d.is_cuda:
            min_rects = min_rects.cuda()
        # calculate minimum rectangle
        min_rects[:, 0] = torch.min(box8pts[:, [0, 2, 4, 6]], dim=1)[0]
        min_rects[:, 1] = torch.min(box8pts[:, [1, 3, 5, 7]], dim=1)[0]
        min_rects[:, 2] = torch.max(box8pts[:, [0, 2, 4, 6]], dim=1)[0]
        min_rects[:, 3] = torch.max(box8pts[:, [1, 3, 5, 7]], dim=1)[0]
        return min_rects

    return box3d_top

def draw_boxes(reg, prob, images, calibs, ids, tag):
    prob = prob.view(cfg.N, -1)
    batch_boxes3d = delta_to_boxes3d(reg, cfg.anchors)
    mask = torch.gt(prob, cfg.score_threshold)
    mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

    for batch_id in range(cfg.N):
        boxes3d = torch.masked_select(batch_boxes3d[batch_id], mask_reg[batch_id]).view(-1, 7)
        scores = torch.masked_select(prob[batch_id], mask[batch_id])

        image = images[batch_id]
        calib = calibs[batch_id]
        id = ids[batch_id]

        if len(boxes3d) != 0:

            boxes3d_corner = box3d_center_to_corner_batch(boxes3d)
            boxes2d = box3d_corner_to_top_batch(boxes3d_corner)
            boxes2d_score = torch.cat((boxes2d, scores.unsqueeze(1)), dim=1)

            # NMS
            keep = pth_nms(boxes2d_score, cfg.nms_threshold)
            boxes3d_corner_keep = boxes3d_corner[keep]
            print("No. %d objects detected" % len(boxes3d_corner_keep))

            rgb_2D = project_velo2rgb(boxes3d_corner_keep, calib)
            img_with_box = draw_rgb_projections(image, rgb_2D, color=(0, 0, 255), thickness=1)
            cv2.imwrite('results/%s_%s.png' % (id,tag), img_with_box)

        else:
            cv2.imwrite('results/%s_%s.png' % (id,tag), image)
            print("No objects detected")













