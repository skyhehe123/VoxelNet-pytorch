from __future__ import division
import numpy as np
from config import config as cfg
import math
import mayavi.mlab as mlab
import cv2
from box_overlaps import *

def get_filtered_lidar(lidar):

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]

    filter_x = np.where((pxs >= cfg.xrange[0]) & (pxs < cfg.xrange[1]))[0]
    filter_y = np.where((pys >= cfg.yrange[0]) & (pys < cfg.yrange[1]))[0]
    filter_z = np.where((pzs >= cfg.zrange[0]) & (pzs < cfg.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    return lidar[filter_xyz]

def lidar_to_top(lidar):

    X0, Xn = 0, cfg.W
    Y0, Yn = 0, cfg.H
    Z0, Zn = 0, cfg.D

    width  = Yn - Y0
    height   = Xn - X0
    channel = Zn - Z0  + 2

    lidar = get_filtered_lidar(lidar)

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]

    qxs=((pxs-cfg.xrange[0])/cfg.vw).astype(np.int32)
    qys=((pys-cfg.yrange[0])/cfg.vh).astype(np.int32)
    qzs=((pzs-cfg.zrange[0])/cfg.vd).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    mask = np.ones(shape=(height,width,channel-1), dtype=np.float32)* -5

    for i in range(len(pxs)):
        top[-qxs[i], -qys[i], -1]= 1+ top[-qxs[i], -qys[i], -1]
        if pzs[i]>mask[-qxs[i], -qys[i],qzs[i]]:
            top[-qxs[i], -qys[i], qzs[i]] = max(0,pzs[i]-cfg.zrange[0])
            mask[-qxs[i], -qys[i],qzs[i]]=pzs[i]
        if pzs[i]>mask[-qxs[i], -qys[i],-1]:
            mask[-qxs[i], -qys[i],-1]=pzs[i]
            top[-qxs[i], -qys[i], -2]=prs[i]


    top[:,:,-1] = np.log(top[:,:,-1]+1)/math.log(64)

    if 1:
        # top_image = np.sum(top[:,:,:-1],axis=2)
        density_image=top[:,:,-1]
        density_image = density_image-np.min(density_image)
        density_image = (density_image/np.max(density_image)*255).astype(np.uint8)
        # top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    return top, density_image

def draw_lidar(lidar, is_grid=False, is_top_region=True, fig=None):

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))

    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)

    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if 1:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = cfg.xrange[0]
        x2 = cfg.xrange[1]
        y1 = cfg.yrange[0]
        y2 = cfg.yrange[1]
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)



    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

    return fig

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,0,0), line_width=2):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        for k in range(0,4):

            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+3)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

def project_velo2rgb(velo,calib):
    T=np.zeros([4,4],dtype=np.float32)
    T[:3,:]=calib['Tr_velo2cam']
    T[3,3]=1
    R=np.zeros([4,4],dtype=np.float32)
    R[:3,:3]=calib['R0']
    R[3,3]=1
    num=len(velo)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for i in range(len(velo)):
        box3d=np.ones([8,4],dtype=np.float32)
        box3d[:,:3]=velo[i]
        M=np.dot(calib['P2'],R)
        M=np.dot(M,T)
        box2d=np.dot(M,box3d.T)
        box2d=box2d[:2,:].T/box2d[2,:].reshape(8,1)
        projections[i] = box2d
    return projections

def draw_rgb_projections(image, projections, color=(255,255,255), thickness=2, darker=1):

    img = image.copy()*darker
    num=len(projections)
    forward_color=(255,255,0)
    for n in range(num):
        qs = projections[n]
        # cv2.putText(image,"%d"%n, (qs[6,0],qs[6,1]), cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            # if length_filter((qs[i,0],qs[i,1]),(qs[j,0],qs[j,1]),img):
            #     break
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k+4,(k+1)%4 + 4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k,k+4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

        cv2.line(img, (qs[3,0],qs[3,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[7,0],qs[7,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[6,0],qs[6,1]), (qs[2,0],qs[2,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[3,0],qs[3,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[3,0],qs[3,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)

    return img

def lidar_to_top_coords(x, y):
    xx = cfg.H - int((y - cfg.yrange[0]) / cfg.vh)
    yy = cfg.W - int((x - cfg.xrange[0]) / cfg.vw)
    return xx, yy

def draw_polygons(image, polygons,color=(255,255,255), thickness=1, darken=1):

    img = image.copy() * darken
    for polygon in polygons:
        tup0, tup1, tup2, tup3 = [lidar_to_top_coords(*tup) for tup in list(zip(polygon[0::2], polygon[1::2]))]
        cv2.line(img, tup0, tup1, color, thickness, cv2.LINE_AA)
        cv2.line(img, tup1, tup2, color, thickness, cv2.LINE_AA)
        cv2.line(img, tup2, tup3, color, thickness, cv2.LINE_AA)
        cv2.line(img, tup3, tup0, color, thickness, cv2.LINE_AA)
    return img

def draw_rects(image, rects, color=(255,255,255), thickness=1, darken=1):

    img = image.copy() * darken
    for rect in rects:
        tup0,tup1 = [lidar_to_top_coords(*tup) for tup in list(zip(rect[0::2], rect[1::2]))]
        cv2.rectangle(img, tup0, tup1, color, thickness, cv2.LINE_AA)
    return img

def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle

def box3d_cam_to_velo(box3d, Tr):

    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle

        return angle

    h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)

def anchors_center_to_corner(anchors):
    N = anchors.shape[0]
    anchor_corner = np.zeros((N, 4, 2))
    for i in range(N):
        anchor = anchors[i]
        translation = anchor[0:3]
        h, w, l = anchor[3:6]
        rz = anchor[-1]
        Box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2]])
        # re-create 3D bounding box in velodyne coordinate system
        rotMat = np.array([
            [np.cos(rz), -np.sin(rz)],
            [np.sin(rz), np.cos(rz)]])
        velo_box = np.dot(rotMat, Box)
        cornerPosInVelo = velo_box + np.tile(translation[:2], (4, 1)).T
        box2d = cornerPosInVelo.transpose()
        anchor_corner[i] = box2d
    return anchor_corner

def box3d_corner_to_top_batch(boxes3d, use_min_rect=False):
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
        box8pts = np.array(box3d_top)
        min_rects = np.zeros((box8pts.shape[0], 4), dtype=np.int32)
        # calculate minimum rectangle
        min_rects[:, 0] = np.min(box8pts[:, [0, 2, 4, 6]], axis=1)
        min_rects[:, 1] = np.min(box8pts[:, [1, 3, 5, 7]], axis=1)
        min_rects[:, 2] = np.max(box8pts[:, [0, 2, 4, 6]], axis=1)
        min_rects[:, 3] = np.max(box8pts[:, [1, 3, 5, 7]], axis=1)
        return min_rects

    return box3d_top

def corner_to_standup_box2d_batch(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)
    return standup_boxes2d

def box3d_corner_to_center_batch(box3d_corner):
    # (N, 8, 3) -> (N, 7)
    assert box3d_corner.ndim == 3
    batch_size = box3d_corner.shape[0]

    xyz = np.mean(box3d_corner[:, :4, :], axis=1)

    h = abs(np.mean(box3d_corner[:, 4:, 2] - box3d_corner[:, :4, 2], axis=1, keepdims=True))

    w = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    l = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    theta = (np.arctan2(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1],
                        box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0]) +
             np.arctan2(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1],
                        box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0]) +
             np.arctan2(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0],
                        box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1]) +
             np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0],
                        box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

    return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(batch_size, 7)

def get_anchor3d(anchors):
    num = anchors.shape[0]
    anchors3d = np.zeros((num,8,3))
    anchors3d[:, :4, :2] = anchors
    anchors3d[:, :, 2] = cfg.z_a
    anchors3d[:, 4:, :2] = anchors
    anchors3d[:, 4:, 2] = cfg.z_a + cfg.h_a
    return anchors3d

def load_kitti_label(label_file, Tr):

    with open(label_file,'r') as f:
        lines = f.readlines()

    gt_boxes3d_corner = []

    num_obj = len(lines)

    for j in range(num_obj):
        obj = lines[j].strip().split(' ')

        obj_class = obj[0].strip()
        if obj_class not in cfg.class_list:
            continue

        # abandon box which are not lie in the range
        if float(obj[-2]) < cfg.xrange[0] or float(obj[-2]) > cfg.xrange[1]:
            continue
        if -float(obj[-4]) < cfg.yrange[0] or -float(obj[-4]) > cfg.yrange[1]:
            continue
        if -float(obj[-3]) < cfg.zrange[0] or -float(obj[-3]) > cfg.zrange[1]:
            continue

        box3d_corner = box3d_cam_to_velo(obj[8:], Tr)

        gt_boxes3d_corner.append(box3d_corner)

    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1,8,3)

    return gt_boxes3d_corner

def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])
    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)
    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)
    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)
    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)
    return points[:, 0:3]

def box_transform(boxes_corner, tx, ty, tz, r=0):
    # boxes_corner (N, 8, 3)
    for idx in range(len(boxes_corner)):
        boxes_corner[idx] = point_transform(boxes_corner[idx], tx, ty, tz, rz=r)
    return boxes_corner

def cal_iou2d(box1_corner, box2_corner):
    box1_corner = np.reshape(box1_corner, [4, 2])
    box2_corner = np.reshape(box2_corner, [4, 2])
    buf1 = np.zeros((cfg.H, cfg.W, 3))
    buf2 = np.zeros((cfg.H, cfg.W, 3))
    buf1 = cv2.fillConvexPoly(buf1, box1_corner, color=(1,1,1))[..., 0]
    buf2 = cv2.fillConvexPoly(buf2, box2_corner, color=(1,1,1))[..., 0]
    indiv = np.sum(np.absolute(buf1-buf2))
    share = np.sum((buf1 + buf2) == 2)
    if indiv == 0:
        return 0.0 # when target is out of bound
    return share / (indiv + share)

def aug_data(lidar, gt_box3d_corner):
    np.random.seed()

    choice = np.random.randint(1, 10)

    if choice >= 7:
        for idx in range(len(gt_box3d_corner)):
            # TODO: precisely gather the point
            is_collision = True
            _count = 0
            while is_collision and _count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal()
                t_y = np.random.normal()
                t_z = np.random.normal()
                # check collision
                tmp = box_transform(
                    gt_box3d_corner[[idx]], t_x, t_y, t_z, t_rz)
                is_collision = False
                for idy in range(idx):
                    iou = cal_iou2d(box3d_corner_to_top_batch(tmp)[0],
                                    box3d_corner_to_top_batch(gt_box3d_corner[[idy]])[0])
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = gt_box3d_corner[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                gt_box3d_corner[idx] = box_transform(
                    gt_box3d_corner[[idx]], t_x, t_y, t_z, t_rz)

        gt_box3d = gt_box3d_corner

    elif choice < 7 and choice >= 4:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        gt_box3d = box_transform(gt_box3d_corner, 0, 0, 0, r=angle)

    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        gt_box3d = gt_box3d_corner * factor

    return lidar, gt_box3d

def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return keep

def test():
    import os
    import glob
    import matplotlib.pyplot as plt

    lidar_path = os.path.join('./data/KITTI/training', "crop/")
    image_path = os.path.join('./data/KITTI/training', "image_2/")
    calib_path = os.path.join('./data/KITTI/training', "calib/")
    label_path = os.path.join('./data/KITTI/training', "label_2/")


    files_list=glob.glob(label_path + '/*.txt')


    file=[i.strip().split('/')[-1][:-4] for i in files_list]

    i=8

    lidar_file = lidar_path + '/' + file[i] + '.bin'
    calib_file = calib_path + '/' + file[i] + '.txt'
    label_file = label_path + '/' + file[i] + '.txt'
    image_file = image_path + '/' + file[i] + '.png'

    image = cv2.imread(image_file)
    print("Processing: ", lidar_file)
    lidar = np.fromfile(lidar_file, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))

    calib = load_kitti_calib(calib_file)
    gt_box3d = load_kitti_label(label_file, calib['Tr_velo2cam'])

    # augmentation
    #lidar, gt_box3d = aug_data(lidar,gt_box3d)

    # filtering
    lidar = get_filtered_lidar(lidar)


    # view in point cloud

    fig = draw_lidar(lidar, is_grid=False, is_top_region=True)
    draw_gt_boxes3d(gt_boxes3d=gt_box3d, fig=fig)
    mlab.show()

    # view in image

    gt_3dTo2D = project_velo2rgb(gt_box3d, calib)
    img_with_box = draw_rgb_projections(image,gt_3dTo2D, color=(0,0,255),thickness=1)
    plt.imshow(img_with_box[:,:,[2,1,0]])
    plt.show()

    # view in bird-eye view

    top_new, density_image=lidar_to_top(lidar)
    gt_box3d_top = box3d_corner_to_top_batch(gt_box3d, use_min_rect=False)
    density_with_box = draw_polygons(density_image,gt_box3d_top)
    plt.imshow(density_with_box,cmap='gray')
    plt.show()

if __name__ == '__main__':
    test()