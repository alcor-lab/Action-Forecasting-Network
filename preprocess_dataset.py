import matplotlib
matplotlib.use('TkAgg') 
import os
import cv2
import numpy as np
import random
import pprint
import time
from poseEstimation import OpenPose
from tqdm import tqdm
import pickle
#from dataset_manager import Dataset
import config
import multiprocessing.dummy as mp
import datetime
import tensorflow as tf
#import json
#import h5py
import numpy as np
import time
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import mask.MaskUCL
from mask.MaskUCL import MaskUCL


class preprocess:
    def __init__(self):
        self.preprocess_path = config.preprocess_path
        with tf.Session() as sess:
            self.sess = sess
            K.set_session(sess)
            self.openpose = OpenPose(self.sess)
            self.openpose.load_openpose_weights()
            self.mask = MaskUCL()
            set_session(self.sess)
            self.mask.loadModel(model='COCO', configDisplay=True)
            self.graph = tf.get_default_graph()
            file_path_list = []
            tot_frames = 0
            for root, dirs, files in os.walk(config.data_path):
                for fl in files:
                    if fl.endswith('.avi'):
                        path = root + '/' + fl      
                        video = cv2.VideoCapture(path)
                        video.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
                        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        tot_frames += length
                        file_path_list.append(path)

            tot_file = len(file_path_list)
            self.pbar_file = tqdm(total=tot_file, leave=False, desc='Tot_video')
            self.pbar_frame = tqdm(total=tot_frames, leave=False, desc='Tot_video')
            pool = mp.Pool(processes=config.processes)
            pool.map(self.multi_file_preprocess, file_path_list)

    def multi_file_preprocess(self, path):
        split_path = path.split('/')
        fl = split_path[-1]
        root = '/'.join(split_path[0:-1]) + '/'
        # print(root)
        path_split = path.split('/') 
        file_path = self.preprocess_path + '/' + path_split[-3] + '/' + path_split[-2] + '/'+fl[:-4]
        # print(root)
        # print(file_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        start_frame = 0
        video = cv2.VideoCapture(path)
        video.set(cv2.CAP_PROP_POS_AVI_RATIO, start_frame)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.set(1, start_frame)
        ret, prev = video.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        # pbar_frame = tqdm(total=length, leave=False, desc='Frame')
        d= 0
        for frame in range(1, length-2):
            d+=1
            frame_path = file_path + "/" + str(frame)
            # if True:
            if not os.path.isfile(frame_path + '_skeleton.jpg'):
                frame_matrix = np.zeros(shape=(368, 368, 11), dtype=float)
                try:
                    video.set(1, frame)
                    ret, im = video.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, flow=None,
                                                        pyr_scale=0.5, levels=1,
                                                        winsize=15, iterations=3,
                                                        poly_n=5, poly_sigma=1.1, flags=0)
                    prev_gray = gray
                    res_im = cv2.resize(im, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                    res_flow = cv2.resize(flow, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                    pafMat, heatMat, skeleton = self.openpose.compute_pose_frame(res_im, estimate_skeleton=True)
                    set_session(self.sess)
                    with self.graph.as_default():
                        resulting_mask, scores = self.mask.detectLorenzo(im, showImage=False)
                    if scores.shape[0] != 0:
                        best_mask = np.argmax(scores) + 3
                        best_person = resulting_mask[:,:,best_mask]
                    else:
                        best_person = res_im[:,:,0]*0
                    res_pafMat = cv2.resize(pafMat, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                    res_heatMat = cv2.resize(heatMat, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                    res_best_mask = cv2.resize(best_person, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                    res_skeleton = cv2.resize(skeleton, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                    frame_matrix[:, :, :3] = cv2.normalize(res_im, None, 0, 255, cv2.NORM_MINMAX)
                    frame_matrix[:, :, 3] = cv2.normalize(res_pafMat, None, 0, 255, cv2.NORM_MINMAX)
                    frame_matrix[:, :, 4] = cv2.normalize(res_heatMat, None, 0, 255, cv2.NORM_MINMAX)
                    frame_matrix[:, :, 5:7] = cv2.normalize(res_flow, None, 0, 255, cv2.NORM_MINMAX)
                    frame_matrix[:, :, 7] = cv2.normalize(res_best_mask, None, 0, 255, cv2.NORM_MINMAX)
                    frame_matrix[:, :, 8:11] = cv2.normalize(skeleton, None, 0, 255, cv2.NORM_MINMAX)
                    frame_matrix = frame_matrix.astype(np.uint8)
                    self.save_frame(frame_matrix, frame_path)
                except Exception as e:
                    print(e)
                    print(path + '    frame:' + str(frame)+ '    tot:' + str(length))
                    pass
            self.pbar_frame.update(1)
            #if d ==3:
                #break
        #pbar_frame.refresh()
        # pbar_frame.close()
        self.pbar_file.update(1)


    def save_frame(self, frame_matrix, frame_path):
        cv2.imwrite(frame_path + '_rgb.jpg',frame_matrix[:, :, :3])
        cv2.imwrite(frame_path + '_pafMat.jpg',frame_matrix[:, :, 3])
        cv2.imwrite(frame_path + '_heatMat.jpg',frame_matrix[:, :, 4])
        cv2.imwrite(frame_path + '_flow_1.jpg',frame_matrix[:, :, 5])
        cv2.imwrite(frame_path + '_flow_2.jpg',frame_matrix[:, :, 6])
        cv2.imwrite(frame_path + '_mask.jpg',frame_matrix[:, :, 7])
        cv2.imwrite(frame_path + '_skeleton.jpg',frame_matrix[:, :, 8:11])

prep = preprocess()
