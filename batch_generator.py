import os
import cv2
import numpy as np
import random
import pprint
import time
from poseEstimation import OpenPose
from tqdm import tqdm
import pickle
from dataset_manager import Dataset
import config
import multiprocessing.dummy as mp


pp = pprint.PrettyPrinter(indent=4)

class IO_manager:
    def __init__(self, sess):
        self.dataset = Dataset()
        self.num_classes = self.dataset.number_of_classes
        self.sess = sess
        self.openpose = None
        self.snow_ball_minimum = config.snow_ball_classes
        if os.path.isfile('dataset/hidden_states_collection.pkl') and config.reuse_HSM:
            with open('dataset/hidden_states_collection.pkl', 'rb') as f:
                self.hidden_states_collection = pickle.load(f)
        else:
            self.hidden_states_collection = {}
        self.hidden_states_statistics()

    def compute_batch(self, pbar, Devices, Train, augment=True):
        def multiprocess_batch(x):
            X, Y, c, h, video_name_collection, segment_collection, next_label, multiple_next_label = self.batch_generator(pbar, Train)
            return {'X': X, 'Y': Y, 'c': c, 'h': h,
                    'next_Y': next_label,
                    'multi_next_Y': multiple_next_label,
                    'video_name_collection': video_name_collection,
                    'segment_collection': segment_collection}

        pool = mp.Pool(processes=config.processes)
        ready_batch = pool.map(multiprocess_batch, range(0, config.tasks*Devices))
        ready_batch = self.add_pose(ready_batch, self.sess, augment)
        ready_batch = self.group_batches(ready_batch, Devices, pbar)
        pbar.close()
        pool.close()
        pool.join()
        return ready_batch

    def group_batches(self, ready_batch, Devices, pbar):
        new_collection = []
        for j in range(config.tasks):
            selected_batches = ready_batch[j: j+Devices]
            if Devices == 'g':
                X = [v['X'] for v in selected_batches]
                X = np.expand_dims(X, axis=0)
                Y = [v['Y'] for v in selected_batches]
                Y = np.expand_dims(Y, axis=0)
                c = [v['c'] for v in selected_batches]
                c = np.expand_dims(c, axis=0)
                h = [v['h'] for v in selected_batches]
                h = np.expand_dims(h, axis=0)
                next_label = [v['next_Y'] for v in selected_batches]
                next_label = np.expand_dims(next_label, axis=0)
                multi_next_label = [v['multi_next_Y'] for v in selected_batches]
                multi_next_label = np.expand_dims(next_label, axis=0)
            else:
                X = [v['X'] for v in selected_batches]
                X = np.stack(X, axis=0)
                Y = [v['Y'] for v in selected_batches]
                Y = np.stack(Y, axis=0)
                c = [v['c'] for v in selected_batches]
                c = np.stack(c, axis=0)
                h = [v['h'] for v in selected_batches]
                h = np.stack(h, axis=0)
                next_label = [v['next_Y'] for v in selected_batches]
                next_label = np.stack(next_label, axis=0)
                multi_next_label = [v['multi_next_Y'] for v in selected_batches]
                multi_next_label = np.stack(next_label, axis=0)
            video_name_collection = []
            segment_collection = []
            for i in range(Devices):
                video_name_collection.append(ready_batch[j+i]['video_name_collection'])
                segment_collection.append(ready_batch[j+i]['segment_collection'])
            new_collection.append({'X': X, 'Y': Y, 'c': c, 'h': h,
                    'next_Y': next_label,
                    'multi_next_Y': multi_next_label,
                    'video_name_collection': video_name_collection,
                    'segment_collection': segment_collection})
            pbar.update(1)
        return new_collection

    def batch_generator(self, pbar, Train=True):
        random.seed(time.time())
        segment_collection = []
        video_name_collection = []
        batch = np.zeros(shape=(config.Batch_size, config.frames_per_step, config.op_input_height, config.op_input_width, 7), dtype=float)
        labels = np.zeros(shape=(config.Batch_size, self.num_classes), dtype=int)
        next_labels = np.zeros(shape=(config.Batch_size, self.num_classes), dtype=int)
        multiple_next_labels = np.zeros(shape=(config.Batch_size, self.num_classes), dtype=int)
        c = np.zeros(shape=(config.Batch_size, config.hidden_states_dim), dtype=float)
        h = np.zeros(shape=(config.Batch_size, config.hidden_states_dim), dtype=float)

        # Selecting correct dataset
        if Train:
            dataset = self.dataset.train_collection
        else:
            dataset = self.dataset.test_collection

        j = 0
        while j < config.Batch_size:
            random_entry = self.entry_selector(dataset)
            current_label = random_entry['now_label']
            next_label = random_entry['next_label']
            multiple_next = random_entry['all_next_label']
            path = random_entry['path']
            segment = random_entry['segment']
            label_history = random_entry['history']
            one_input, frame_list = self.extract_one_input(path, segment, pbar)
            config.snow_ball_step_count += 1

            segment_collection.append(segment)
            video_name_collection.append(path)
            batch[j, :, :, :, :] = one_input
            labels[j, current_label] = 1
            next_labels[j, next_label] = 1
            for element in multiple_next:
                multiple_next_labels[j, element] = 1
            if path in self.hidden_states_collection:
                start_frame = segment_collection[j][0] - 1
                if start_frame in self.hidden_states_collection[path].keys():
                    c[j, :] = self.hidden_states_collection[path][start_frame]['c']
                    h[j, :] = self.hidden_states_collection[path][start_frame]['h']

            j = j + 1

        pbar.update(1)
        pbar.refresh()
        return batch, labels, c, h, video_name_collection, segment_collection, next_labels, multiple_next_labels

    def entry_selector(self, dataset):
        random.seed(time.time())
        random_couple = random.choice(list(dataset))
        entry = random.choice(dataset[random_couple])
        return entry

    def label_calculator(self, frame_list, path, untrimmed):
        label_clip = {}
        for frame in frame_list:
            if frame not in untrimmed[path]:
                print(frame_list)
                print(frame_list)
            label = untrimmed[path][frame]
            if label not in label_clip:
                label_clip[label] = 0
            label_clip[label] += 1
        try:
            final_label = max(label_clip, key=label_clip.get)
        except Exception as e:
            print(label_clip, frame_list)
            final_label = 0
            pass
        return final_label

    def extract_one_input(self, video_path, segment, pbar):
        one_input = np.zeros(shape=(config.frames_per_step, config.op_input_height, config.op_input_width, 7), dtype=float)
        extracted_frames = {}
        frame_list = []
        try:
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            linspace_frame = np.linspace(segment[0], segment[1], num=config.frames_per_step)
            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if (linspace_frame[-1] == (tot_frames-1)):
                linspace_frame[-1] -= 1
            if (linspace_frame[-1] == (tot_frames-2)):
                linspace_frame[-1] -= 2
            z = 0
            for frame in linspace_frame:
                try:
                    frame = int(frame)
                    frame_prev = frame - 1

                    # Extracting Frames from video: if frame has been already extracted
                    # then take it from extracted frame collection.

                    if frame in extracted_frames:
                        im = extracted_frames[frame]['im']
                        gray = extracted_frames[frame]['gray']
                    else:
                        video.set(1, frame)
                        ret, im = video.read()
                        if not ret:
                            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                            print('\n', ret)
                            print('\nframe', frame)
                            print('\ntotframe', tot_frames)
                            print('\nvideo_path', video_path)
                            print('\nsegment', segment)
                        im = cv2.resize(im, dsize=(config.op_input_height, config.op_input_width), interpolation=cv2.INTER_CUBIC)
                        extracted_frames[frame] = {}
                        extracted_frames[frame]['im'] = im
                        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        extracted_frames[frame]['gray'] = gray

                    if frame_prev in extracted_frames:
                        im_prev = extracted_frames[frame_prev]['im']
                        gray_prev = extracted_frames[frame_prev]['gray']
                    else:
                        video.set(1, frame_prev)
                        ret, im_prev = video.read()
                        if not ret:
                            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                            print('\n', ret)
                            print('\nprevframe', frame)
                            print('\ntotframe', tot_frames)
                            print('\nvideo_path', video_path)
                            print('\nsegment', segment)
                        im_prev = cv2.resize(im_prev, dsize=(config.op_input_height, config.op_input_width), interpolation=cv2.INTER_CUBIC)
                        extracted_frames[frame_prev] = {}
                        extracted_frames[frame_prev]['im'] = im_prev
                        gray_prev = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        extracted_frames[frame_prev]['gray'] = gray_prev

                    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray, flow=None,
                                                        pyr_scale=0.5, levels=1,
                                                        winsize=15, iterations=3,
                                                        poly_n=5, poly_sigma=1.1, flags=0)
                    norm_flow = flow
                    norm_flow = cv2.normalize(flow, norm_flow, 0, 255, cv2.NORM_MINMAX)
                    one_input[z, :, :, :3] = im
                    one_input[z, :, :, 5:7] = flow
                    z += 1
                except Exception as e:
                    pass
                pbar.update(1)
        except Exception as e:
            pass

        frame_list = extracted_frames.keys()
        extracted_frames = None
        return one_input, frame_list

    def add_pose(self, X, sess, augment=True):
        shape = (X[0])['X'].shape
        total = len(X) * shape[0] * shape[1]
        augment = 'none'
        if augment:
            augment = random.choice(['none', 'horizzontal', 'vertical', 'both'])
        pbar = tqdm(total=total, leave=False, desc='Computing Poses')
        for k in range(len(X)):
            X_data = (X[k])['X']
            shape = X_data.shape
            shrinked_X = np.zeros(shape=(shape[0], shape[1], config.out_H, config.out_W, shape[4]), dtype=float)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    try:
                        pafMat, heatMat = self.openpose.compute_pose_frame(X_data[i, j, :, :, :3])
                        X_data[i, j, :, :, 3] = heatMat
                        X_data[i, j, :, :, 4] = pafMat
                        final_frame = X_data[i, j, :, :, :]
                        shrinked_frame = cv2.resize(final_frame, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
                        shrinked_frame = self.augment_data(shrinked_frame, augment)
                        shrinked_X[i, j, :, :, :] = shrinked_frame
                        pbar.update(1)
                    except Exception as e:
                        pass
            if config.show_pic:
                self.show_input_pic(X_data)
            X[k]['X'] = shrinked_X
        pbar.refresh()
        pbar.clear()
        pbar.close()
        return X

    def show_input_pic(self,X_data):
        shape = X_data.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                heatMat = cv2.normalize(X_data[i, j, :, :, 3],None,0,255,cv2.NORM_MINMAX)
                pafMat = cv2.normalize(X_data[i, j, :, :, 4],None,0,255,cv2.NORM_MINMAX)
                im = X_data[i, j, :, :, :3]/255
                flow = X_data[i, j, :, :, 5:]
                mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
                hsv = cv2.cvtColor(im.astype('uint8'), cv2.COLOR_RGB2HSV)
                hsv[:,:,0] = ang * (180/ np.pi / 2)
                hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                hsv = np.asarray(hsv, dtype= np.float32)
                heatMat = np.asarray(heatMat, dtype= np.float32)
                pafMat = np.asarray(pafMat, dtype= np.float32)
                rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                pafMat_grey = cv2.cvtColor(pafMat, cv2.COLOR_GRAY2BGR)
                heatMat_grey = cv2.cvtColor(heatMat, cv2.COLOR_GRAY2BGR)
                print(im.shape,pafMat_grey.shape,heatMat_grey.shape,rgb_flow.shape)
                numpy_horizontal_concat = np.concatenate((im, pafMat_grey/255), axis=1)
                numpy_horizontal_concat_2 = np.concatenate((numpy_horizontal_concat, heatMat_grey/255), axis=1)
                numpy_horizontal_concat_3 = np.concatenate((numpy_horizontal_concat_2, rgb_flow), axis=1)
                cv2.imshow('ImageWindow',numpy_horizontal_concat_3)
                cv2.waitKey()

    def save_hidden_state_collection(self):
        with open('dataset/hidden_states_collection.pkl', 'wb') as f:
            pickle.dump(self.hidden_states_collection, f, pickle.HIGHEST_PROTOCOL)

    def add_hidden_state(self, video_name, frame, h, c):
        if video_name not in self.hidden_states_collection:
            self.hidden_states_collection[video_name] = {}
        if frame not in self.hidden_states_collection[video_name]:
                self.hidden_states_collection[video_name][frame] = {}
                self.hidden_states_collection[video_name][frame]['number_of_visits'] = 0
        self.hidden_states_collection[video_name][frame]['h'] = h
        self.hidden_states_collection[video_name][frame]['c'] = c
        self.hidden_states_collection[video_name][frame]['number_of_visits'] += 1

    def hidden_states_statistics(self):
        number_of_visits = 0
        second_analysed = 0
        for video in self.hidden_states_collection:
            second_analysed = second_analysed + len(self.hidden_states_collection[video])
            for frame in self.hidden_states_collection[video]:
                number_of_visits = number_of_visits + self.hidden_states_collection[video][frame]['number_of_visits']
        print('\n\tAnalysed Seconds: ' + str(second_analysed) + '\n\tNumber of visits: ' + str(number_of_visits))

    def snow_ball_labels_calculator(self):
        number_of_classes = int(config.snow_ball_step_count / config.snow_ball_per_class)
        if number_of_classes > self.snow_ball_minimum:
            self.snow_ball_minimum = number_of_classes
        possible_label = list(range(1, self.snow_ball_minimum+1))
        return possible_label

    def start_openPose(self):
        self.openpose = OpenPose(self.sess)

    def augment_data(self, frame, type):
        if type is 'none':
            return frame
        elif type is 'horizzontal':
            return cv2.flip( frame, 0 )
        elif type is 'vertical':
            return cv2.flip( frame, 1 )
        elif type is 'both':
            return cv2.flip( frame, -1 )
