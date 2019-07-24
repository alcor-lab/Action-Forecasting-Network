import tensorflow as tf
import os
import multiprocessing.dummy as mp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pprint
from batch_generator import IO_manager
from network import activity_network
import cv2
import pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

c3d_saved_weigths = "sports1m_finetuning_ucf101.model"
c3d_meta = "c3d.pb"
number_of_classes = 16
C3d_number_of_frames = 16
C3d_height = 112
C3d_weigth = C3d_height
pp = pprint.PrettyPrinter(indent=4)

mode = tf.estimator.ModeKeys.TRAIN

config = tf.ConfigProto(inter_op_parallelism_threads=7)
config.gpu_options.allow_growth = True

learning_rate = 0.1
gradient_clipping_norm = 1.0
c3d_dropout = 0.6
preLstm_dropout = 0.6
Lstm_dropout = 0.6
C3d_CHANNELS = 7
C3d_Output_features = 250
lstm_units = C3d_Output_features * 2

Batch_size = 30
frames_per_step = 6
window_size = 1
height = 112
width = height * 1

load_c3d = True
use_pretrained_model = True
model_filename = './checkpoint/Net_weigths.model'
Tot_steps = 1000000

tasks = 7

def test():
    reuse = False
    if os.path.isfile('distribution_results/activity.pkl') and reuse:
        with open('distribution_results/activity.pkl', 'rb') as f:
            video_label = pickle.load(f)
    else:
        video_label = {}

    test_video_path = 'dataset/Video/BreakfastII_15fps_qvga_sync'

    with tf.Session(config=config) as sess:
        Network = activity_network(number_of_classes=49)
        IO_tool = IO_manager(Batch_size, frames_per_step, window_size, sess)
        IO_tool.openpose.load_openpose_weights()
        IO_tool.hidden_states_dim = Network.lstm_units
        sess.run(Network.init)
        dataset = IO_tool.dataset.Train_dataset
        untrimmed = IO_tool.dataset.untrimmed_train_dataset

        train_writer = tf.summary.FileWriter("logdir/test", sess.graph)

        tot_second = 0
        for root, dirs, files in os.walk(test_video_path):
            for video_name in files:
                path = root + '/' + video_name
                if path not in untrimmed.keys():
                    continue
                video = cv2.VideoCapture(path)
                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                max_len = int(video.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                tot_second += max_len

        # Loading initial C3d or presaved network
        use_pretrained_model = True
        if os.path.isfile('./checkpoint/checkpoint') and use_pretrained_model:
            print('new model loaded')
            Network.model_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
        elif load_c3d and use_pretrained_model:
            print('original c3d loaded')
            Network.c3d_loader.restore(sess, c3d_saved_weigths)

        with tf.name_scope('whole_saver'):
            whole_saver = tf.train.Saver()
        whole_saver.save(sess, model_filename, global_step=Network.global_step)

        pbar_whole = tqdm(total=(tot_second), desc='Videos')
        for root, dirs, files in os.walk(test_video_path):
            for video_name in files:
                print(video_name)
                path = root + '/' + video_name
                if path not in untrimmed.keys():
                    continue
                video = cv2.VideoCapture(path)
                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                max_len = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                fps = video.get(cv2.CAP_PROP_FPS)

                step_times = [v for v in range(1, int(max_len / 1))]

                pbar_video = tqdm(total=(len(step_times)), desc='Video_completition')
                for t_end in step_times:
                    if video_name in video_label.keys():
                        if t_end in video_label[video_name].keys():
                            pbar_video.update(1)
                            continue

                    segment =[(t_end - 1)*fps + 2, t_end*fps]

                    pbar = tqdm(total=(tasks * Batch_size * frames_per_step + tasks), leave=False, desc='Batch Generation')

                    def multiprocess_batch(x):
                        X, c, h, label = IO_tool.test_generator(pbar, path, segment)
                        return {'X': X, 'c': c, 'h': h, 'label': label}

                    pool = mp.Pool(processes=tasks)
                    ready_batch = pool.map(multiprocess_batch, range(0, 1))
                    pbar.close()
                    pool.close()
                    pool.join()
                    batch = IO_tool.add_pose(ready_batch, sess, augment=False)[0]

                    conv3 = sess.run([Network.conv3], feed_dict={Network.input_batch: batch['X'],
                                                                 Network.c_input: batch['c'],
                                                                 Network.h_input: batch['h']})

                    if video_name not in video_label.keys():
                        video_label[video_name] = {}

                    video_label[video_name][t_end] = {}
                    video_label[video_name][t_end]['tensor'] = conv3
                    video_label[video_name][t_end]['label'] = batch['label']
                    video_label[video_name][t_end]['limb'] = batch['X'][0, :, :, :, 3]
                    video_label[video_name][t_end]['joint'] = batch['X'][0, :, :, :, 4]

                    pbar_video.update(1)
                    pbar_whole.update(1)

                with open('distribution_results/activity' + video_name + '.pkl', 'wb') as f:
                    pickle.dump(video_label, f, pickle.HIGHEST_PROTOCOL)
                pbar_video.close()


test()
