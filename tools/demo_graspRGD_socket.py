#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

import threading
import socket
import struct
import time
import cv2.aruco as aruco


from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import scipy
from shapely.geometry import Polygon

pi     = scipy.pi
dot    = scipy.dot
sin    = scipy.sin
cos    = scipy.cos
ar     = scipy.array

frame_current=[]
pred_x, pred_y = np.asarray([1,50,100,200]), np.asarray([50, 100, 200, 400])
CLASSES = ('__background__',
           'angle_01', 'angle_02', 'angle_03', 'angle_04', 'angle_05',
           'angle_06', 'angle_07', 'angle_08', 'angle_09', 'angle_10',
           'angle_11', 'angle_12', 'angle_13', 'angle_14', 'angle_15',
           'angle_16', 'angle_17', 'angle_18', 'angle_19')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res50': ('res50_faster_rcnn_iter_240000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'grasp': ('train',)}

ANGLES = (-1000,
           0, -10, -20, -30, -40,
           -50, -60, -70, -80, -90,
           80, 70, 60, 50, 40,
           30, 20, 10, 0)


def Rotate2D(pts,cnt,ang=scipy.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return dot(pts-cnt,ar([[cos(ang),sin(ang)],[-sin(ang),cos(ang)]]))+cnt

def vis_detections(ax, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        #ax.add_patch(
        #    plt.Rectangle((bbox[0], bbox[1]),
        #                  bbox[2] - bbox[0],
        #                  bbox[3] - bbox[1], fill=False,
        #                  edgecolor='red', linewidth=3.5)
        #    )

        # plot rotated rectangles
        pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
        cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
        angle = int(class_name[6:])
        r_bbox = Rotate2D(pts, cnt, -pi/2-pi/20*(angle-1))
        pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
        pred_x, pred_y = pred_label_polygon.exterior.xy

        plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[1:3],pred_y[1:3], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[3:5],pred_y[3:5], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)

        #ax.text(bbox[0], bbox[1] - 2,
        #        '{:s} {:.3f}'.format(class_name, score),
        #        bbox=dict(facecolor='blue', alpha=0.5),
        #        fontsize=14, color='white')

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    #plt.axis('off')
    #plt.tight_layout()

    #save result
    #savepath = './data/demo/results/' + str(image_name) + str(class_name) + '.png'
    #plt.savefig(savepath)

    #plt.draw()

def compute_imgRot(frame):
    # aim to find a ARUCO marker and compute the camera rotation on XY plane

    # camera matrix
    markerLength = 0.06

    # old camera matrix used by Yufeng
    # cameraMatrix = np.array([[297.47608, 0.0, 320], [0.0, 297.14815, 240], [0.0, 0.0, 1.0]])  # camera frame from socket
    # distCoeffs = np.array([0.15190073, -0.8267655, 0.00985276, -0.00435892, 1.58437205])  # fake value

    # new camera matrix obtained at 01/2018
    cameraMatrix = np.array(
        [[592.90077, 0.0, 327.06503], [0.0, 591.07515, 239.40367], [0.0, 0.0, 1.0]])  # camera frame from socket
    distCoeffs = np.array([-0.02067, 0.06351, -0.00285, 0.00083, 0.00000])  # fake value

    # find ARUCO marker
    gray = frame * 255
    # print(gray)
    gray = gray.astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # print (gray)
    # gray = gray.astype(np.uint8)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters_create()  #P_reshape = np.reshape(P, (7*7*2*2))
  #top = P_reshape.argsort()[-10:][::-1]
  #index = top[3]
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print('detected aruco mark (table angle compensation): ' + str(ids))
    gray = aruco.drawDetectedMarkers(gray, corners)

    dst = 0
    tvec = 0
    T = 0
    angle = 0.0
    if ids is not None:
        point0 = corners[0][0][0]
        point3 = corners[0][0][3]
        #if point0[1] >= point3[1]: # rotate counter-clock wise
        angle = np.arctan((point0[1] - point3[1]) / (point3[0] - point0[0]))
        print('counter-clock rotate angle in degree: ' + str(angle/3.14*180)) 
        print('(simply added to detected degree to compensate the table)')
        return angle/3.14*180
        # else: # rotate clock wise
        #     angle = np.arctan((point3[1] - point0[1]) / (point3[0] - point0[0]))
        #     print('clock')
        #     print(angle/3.14*180)
    else:
         return angle

def coordinate_img2table(frame, u, v, rot):
    """project found u v coordinate on image to x y coordinate on table with ARUCO marker."""

    # camera matrix
    markerLength = 0.06

    # old camera matrix used by Yufeng
    #cameraMatrix = np.array([[297.47608, 0.0, 320], [0.0, 297.14815, 240], [0.0, 0.0, 1.0]])  # camera frame from socket
    #distCoeffs = np.array([0.15190073, -0.8267655, 0.00985276, -0.00435892, 1.58437205])  # fake value

    # new camera matrix obtained at 01/2018
    cameraMatrix = np.array([[592.90077, 0.0, 327.06503], [0.0, 591.07515, 239.40367], [0.0, 0.0, 1.0]])  # camera frame from socket
    distCoeffs = np.array([-0.02067,   0.06351,   -0.00285,   0.00083, 0.00000])  # fake value

    # find ARUCO marker
    gray = frame*255
    #print(gray)
    gray = gray.astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    #print (gray)
    #gray = gray.astype(np.uint8)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print('detected aruco mark (table location): ' + str(ids))
    gray = aruco.drawDetectedMarkers(gray, corners)

    dst = 0
    tvec = 0
    T = 0
    point_new = np.zeros((3, 1))
    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], markerLength, cameraMatrix, distCoeffs)

        dst, jacobian = cv2.Rodrigues(rvec)
        T = np.zeros((4, 4))
        T[:3, :3] = dst
        T[:3, 3] = tvec
        T[3, :] = np.array([0, 0, 0, 1])

        # projection
        imagePts = np.array([u, v, 1])
        normal_old = np.array([0, 0, 1])
        ray_center = np.array([0, 0, 0])
        distance_old = 0

        normal_new = np.dot(dst, normal_old) #(3,)
        normal_new = np.expand_dims(normal_new, 1)  #(3,1)
        translation_old = tvec #(1,1,3)
        translation_old = np.squeeze(translation_old, 0) #(1,3)
        distance_new = -(distance_old + np.dot(translation_old, normal_new))

        ray = np.dot(np.linalg.inv(cameraMatrix), imagePts) #(3,)
        t = -(np.dot(normal_new.transpose(), ray_center) + distance_new) / np.dot(normal_new.transpose(), ray) # (1,1)
        intersection = np.multiply(ray, t) #(1,3)
        intersection_homo = np.array([intersection[0,0], intersection[0,1], intersection[0,2], 1])

        point_new = np.dot(np.linalg.inv(T), intersection_homo)

    #print(point_new[0] + 0.36)#0.3
    #print(point_new[1] + 0.35)#0.4#
    print('location on table: ' + str(point_new[0] + 0.30))#0.3
    print('location on table: ' + str(point_new[1] + 0.30))#0.4#
    return point_new[0] + 0.30, point_new[1] + 0.30, rot

def demo_process(sess, net):
    """Detect object classes in an image using pre-computed object proposals."""
    count = 0
    bbs_array = np.array([], dtype=np.float32).reshape(0, 5)
    im = []
    tmp_g = []
    scores= []
    boxes = []
    #fig, ax = plt.subplots(figsize=(12, 12))
    while True:
      if frame_current != []:
        print('\n')
        print('############ Start detection on a new frame ############')
        bbs_array = np.array([], dtype=np.float32).reshape(0, 5)
        # Load the demo image
        #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
        #im = cv2.imread(im_file)
        #print (frame_current)

        im = frame_current
        tmp_g = im[:,:,1]
        im[:,:,1] = im[:,:,2] 
        im[:,:,2] = tmp_g  
        im = im*255

        #img = im.astype('uint8')
        #ax.imshow(img)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(sess, net, im)

        #scores_max = scores[:,1:-1].max(axis=1)
        #scores_max_idx = np.argmax(scores_max)
        #scores = scores[scores_max_idx:scores_max_idx+1,:]
        #boxes = boxes[scores_max_idx:scores_max_idx+1, :]

        timer.toc()
        print('deepGrasp detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

        #fig, ax = plt.subplots(figsize=(12, 12)) #uncommand this when writing files into disk
        # Visualize detections for each class
        CONF_THRESH = 0.001
        CONF_THRESH = 0.1 # demo for spoon 2018/08
        NMS_THRESH = 0.3
 
        top_score = 0;
        top_cls = 0;
        angle_compensated = 0.0
        top_boxes = np.zeros(2)
        top_boxes_coor = np.zeros(4)
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            # stack all det > threshold
            dets_th = np.delete(dets, np.where(dets[:, -1] < CONF_THRESH)[0], axis=0)
            bbs_array = np.vstack((bbs_array, dets_th))

            #vis_detections(ax, im, cls, dets, thresh=CONF_THRESH) #uncommand if you want to visualize
              
            if ( max(cls_scores) > top_score):
              #print (max(cls_scores))
              #print(np.amax(cls_scores))
              #print (np.argmax(cls_scores))
              #print (cls_boxes[np.argmax(cls_scores),:])

              top_boxes_coor = cls_boxes[np.argmax(cls_scores),:]
              top_boxes[0] = (top_boxes_coor[0] + top_boxes_coor[2]) / 2
              top_boxes[1] = (top_boxes_coor[1] + top_boxes_coor[3]) / 2
              top_score = max(cls_scores)
              top_cls = cls_ind

        x_table = 0
        y_table = 0

        if bbs_array.shape[0] != 0:
          print('candadiates grasp above threshold: ' + str(CONF_THRESH) + ' are found!')
          bbs_array_cnt = np.transpose(np.vstack(( bbs_array[:,0]/2+bbs_array[:,2]/2,  bbs_array[:,1]/2+bbs_array[:,3]/2)))
          bbs_array_cnt_mean = bbs_array_cnt.mean(axis=0, keepdims=True)
          bbs_array_dist = np.sum(np.square(bbs_array_cnt - bbs_array_cnt_mean), axis=1)
          bbs_array_ins = np.argmin(bbs_array_dist)
          #circle2 = plt.Circle((bbs_array[bbs_array_ins,0]/2+bbs_array[bbs_array_ins,2]/2, bbs_array[bbs_array_ins,1]/2+bbs_array[bbs_array_ins,3]/2), 2, color='g')
          #ax.add_artist(circle2)

          USE_AVERAGE = False
          if USE_AVERAGE:
              # if need top 10, use this:
              x_table, y_table, _ = coordinate_img2table(frame_current, bbs_array[bbs_array_ins,0]/2+bbs_array[bbs_array_ins,2]/2, bbs_array[bbs_array_ins,1]/2+bbs_array[bbs_array_ins,3]/2, top_cls)
              bbox = top_boxes_coor
          else:
              # if need top 1, use this
              x_table, y_table, _ = coordinate_img2table(frame_current, top_boxes[0], top_boxes[1], top_cls)
              bbox = bbs_array[bbs_array_ins, :4] # (1, 5) = (x_min, y_min, x_max, y_max, score)
          
          angle_compensated = compute_imgRot(frame_current)

          ## plot grasp, need to use change global pred_x and pred_y
          pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
          cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
          r_bbox = Rotate2D(pts, cnt, -pi/2-pi/20*(top_cls-1))
          pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
          global pred_x
          global pred_y
          pred_x, pred_y = pred_label_polygon.exterior.xy

        else:
            print('no candidates above threshold: ' + str(CONF_THRESH) + 'found, lower down threshold')

        print("top class is: " + str(top_cls))
        print("top class angle is: " + str(ANGLES[top_cls] + angle_compensated))
        print("top class location is: " + str(top_boxes[0]) + " " + str(top_boxes[1]))

        x_table_center, y_table_center, _ = coordinate_img2table(frame_current, 320, 240, 0)
        print('camera is facing to '+str(x_table_center)+' '+str(y_table_center)+' on the table') 

        #circle1 = plt.Circle((top_boxes[0], top_boxes[1]), 2, color='y')
        #ax.add_artist(circle1)

        #plt.axis('off')
        #plt.tight_layout()

        #save result
        count = count +1
        #savepath = './data/demo/results_all_cls/' + str(count) + '.png'
        #plt.savefig(savepath)
        #plt.cla()

    
        file = open('/home/fujenchu/projects/robotArm/toy-opencv-mat-socket-server-master_pcl/bbs/rotation.txt', "w")
        file.write(str(x_table) + '\n')
        file.write(str(y_table) + '\n')
        file.write(str(ANGLES[top_cls] + angle_compensated) + '\n')
        file.close()

        file = open('/home/fujenchu/projects/robotArm/toy-opencv-mat-socket-server-master_pcl/bbs/cam_facing.txt', "w")
        file.write(str(x_table_center) + '\n')
        file.write(str(y_table_center) + '\n')
        #file.write(str(ANGLES[top_cls] + angle_compensated) + '\n')
        file.close()


        #plt.draw()
        #plt.show()
        #plt.clf()

        #cv2.imshow('deepGrasp_top_score', frame_current)
        #choice = cv2.waitKey(20)
        #if choice == 27:
        #    break


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    #tfconfig = tf.ConfigProto(device_count={'GPU': 0})

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'res50':
        net = resnetv1(batch_size=1, num_layers=50)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 20,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    demo_process = threading.Thread(target=demo_process, args=(sess,net))
    demo_process.start()

    # TCP/IP
    HOST=''
    PORT=2330
    #PORT = 2325

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    client,addr=s.accept()

    #count = 0
    frame = np.zeros((480, 640, 3))
    while True:
        start = time.time()
        frame = np.zeros((480, 640, 3))
        show_frame = np.zeros((480, 640, 3))
        frame = np.reshape(frame, (480*640*3))
        i=0
        while i < 480*640*3:
            data = client.recv(640*480*3 - i)
            datalen = len(data)
            #if len(data) != 1024:
            datalen_str = str(len(data))
            datalen_str = datalen_str + 'B'
            data_up = struct.unpack(datalen_str, data)
            data_up_np = np.asarray(data_up)
            frame[i:i + len(data)] = data_up_np

            i += len(data)

        # frame_depth = np.zeros((240, 320, 1))
        # show_frame_depth = np.zeros((240, 320, 1))
        # frame_depth = np.reshape(frame_depth, (320*240*1))
        # i=0
        # while i < 240*640*1:
        #     data = client.recv(240*640*1 - i)
        #     # datalen = len(data)
        #     # print(datalen)
        #     # #if len(data) != 1024:
        #     #datalen_str = str(len(data)/2)
        #     #datalen_str = datalen_str + 'H'
        #     #data_up = struct.unpack(datalen_str, data)
        #     #data_up_np = np.asarray(data_up)
        #     data_up_np = np.fromstring(data, dtype='>H')
        #     frame_depth[i:i + 76800] = data_up_np
        #
        #     i += len(data)

        ########## save images for calibration #######################
        # if cv2.waitKey(10) == ord('s'):
        #   count = count +1
        #   savepath = './data/demo/live/' + str(count) + '.png'
        #   cv2.imwrite(savepath, np.reshape(frame, (480, 640, 3)))
        #
        #   savepath = './data/demo/live/' + str(count) + '_d.png'
        #   cv2.imwrite(savepath, np.reshape(frame_depth, (240, 320, 1))/ np.max(frame_depth))
        ##############################################################

        frame_current = np.reshape(frame, (480, 640, 3))/255.0
        #frame_current_depth = 1-(np.reshape(frame_depth, (240, 320, 1)) / np.max(frame_depth))

        duration = time.time()-start
        #print("processed time main =" + str(duration))

        color = (255/255.0, 255/255.0, 255/255.0) 
        thickness = 1

        start_point, end_point = (int(pred_x[0]), int(pred_y[0])), (int(pred_x[1]), int(pred_y[1])) 
        cv2.line(frame_current, start_point, end_point, color, thickness) 

        start_point, end_point = (int(pred_x[2]), int(pred_y[2])), (int(pred_x[3]), int(pred_y[3])) 
        cv2.line(frame_current, start_point, end_point, color, thickness) 

        color = (0, 0, 255/255.0) 
        thickness = 2

        start_point, end_point = (int(pred_x[1]), int(pred_y[1])), (int(pred_x[2]), int(pred_y[2])) 
        cv2.line(frame_current, start_point, end_point, color, thickness) 

        start_point, end_point = (int(pred_x[3]), int(pred_y[3])), (int(pred_x[0]), int(pred_y[0])) 
        cv2.line(frame_current, start_point, end_point, color, thickness) 

        cv2.imshow('frame',frame_current)
        #cv2.imshow('frame_depth', frame_current_depth)
        cv2.waitKey(1)



    #im_names = ['rgd_0076Cropped320.png','rgd_0095.png','pcd0122r_rgd_preprocessed_1.png','pcd0875r_rgd_preprocessed_1.png','resized_0875_2.png']
    #im_names = ['pcd0875r_rgd_preprocessed_1.png','pic_0010.png']
    #for im_name in im_names:
    #    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #    print('Demo for data/demo/{}'.format(im_name))
    #    demo(sess, net, im_name)

    #plt.show()
