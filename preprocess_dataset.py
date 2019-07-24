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
#import multiprocessing.dummy as mp
import datetime
import tensorflow as tf
#import json
#import h5py
import numpy as np
import time


class preprocess:
    def __init__(self):
        self.preprocess_path = "dataset/preprocessed"
        with tf.Session() as sess:
            self.sess = sess
            self.openpose = OpenPose(self.sess)
            self.openpose.load_openpose_weights()
            #json_data = open(config.ocado_annotation).read()
            #dataset = json.loads(json_data)
            #pbar_file = tqdm(total=len(dataset), leave=False, desc='Files')
            for root, dirs, files in os.walk(config.kit_path):
                for fl in files:
                    path = root + '/' + fl
                    folder = root.split('/')[-2:]
                    print(root)
                    if fl.endswith('.avi'):
                        file_path = self.preprocess_path + '/'+fl[:-4]
                        print(file_path)
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)
                        start_frame = 0
                        video = cv2.VideoCapture(path)
                        video.set(cv2.CAP_PROP_POS_AVI_RATIO, start_frame)
                        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.set(1, start_frame)
                        ret, prev = video.read()
                        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
                        pbar_frame = tqdm(total=length, leave=False, desc='Frame')
                        d= 0

                        for frame in range(1, length):
                            d+=1
                            frame_path = file_path + "/" + str(frame)
                            if not os.path.isfile(frame_path + '_heatMat.jpg'):
                                frame_matrix = np.zeros(shape=(368, 368, 7), dtype=float)
                                try:
                                    video.set(1, frame)
                                    ret, im = video.read()
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
                                    pafMat, heatMat = self.openpose.compute_pose_frame(res_im)
                                    res_pafMat = cv2.resize(pafMat, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                                    res_heatMat = cv2.resize(heatMat, dsize=(368, 368), interpolation=cv2.INTER_CUBIC)
                                    frame_matrix[:, :, :3] = cv2.normalize(res_im, None, 0, 255, cv2.NORM_MINMAX)
                                    frame_matrix[:, :, 5:7] = cv2.normalize(res_flow, None, 0, 255, cv2.NORM_MINMAX)
                                    frame_matrix[:, :, 3] = cv2.normalize(res_pafMat, None, 0, 255, cv2.NORM_MINMAX)
                                    frame_matrix[:, :, 4] = cv2.normalize(res_heatMat, None, 0, 255, cv2.NORM_MINMAX)
                                    frame_matrix = frame_matrix.astype(np.uint8)
                                    self.save_frame(frame_matrix, frame_path)
                                except Exception as e:
                                    print(e)
                                    print(path + '    frame:' + str(frame))
                                    pass
                            pbar_frame.update(1)
                            #if d ==3:
                                #break
                        #pbar_frame.refresh()
                        pbar_frame.close()
                        #pbar_file.update(1)
                #pbar_file.refresh()
                #pbar_file.clear()
                #pbar_file.close()

    def save_frame(self, frame_matrix, frame_path):
        cv2.imwrite(frame_path + '_rgb.jpg',frame_matrix[:, :, :3])
        cv2.imwrite(frame_path + '_flow_1.jpg',frame_matrix[:, :, 5])
        cv2.imwrite(frame_path + '_flow_2.jpg',frame_matrix[:, :, 6])
        cv2.imwrite(frame_path + '_pafMat.jpg',frame_matrix[:, :, 3])
        cv2.imwrite(frame_path + '_heatMat.jpg',frame_matrix[:, :, 4])
        # flow = frame_matrix[:, :, 5:7]
        # flow = flow.astype(np.float32)
        # mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
        # hsv = np.zeros_like(frame_matrix[:, :, :3])
        # hsv[:,:,0] = ang *180/np.pi/2
        # hsv[:,:,1] = 255
        # hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        # cv2.imwrite(frame_path + '_hsv_flow.jpg',bgr)

prep = preprocess()
