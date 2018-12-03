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


pp = pprint.PrettyPrinter(indent=4)


class IO_manager:
    def __init__(self, sess):
        self.dataset = Dataset()
        self.num_classes = self.dataset.number_of_classes
        self.sess = sess
        self.openpose = None

    def start_openPose(self):
        self.openpose = OpenPose(self.sess)

        if os.path.isfile('dataset/hidden_states_collection.pkl'):
            with open('dataset/hidden_states_collection.pkl', 'rb') as f:
                self.hidden_states_collection = pickle.load(f)
        else:
            self.hidden_states_collection = {}
        self.hidden_states_statistics()

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
        size = 49
        number_of_classes = int(config.snow_ball_step_count / config.snow_ball_per_class) + 3
        if number_of_classes > size:
            size = number_of_classes
        return range(1, size)

    def augment_data(self, frame, type):
        if type is 'none':
            return frame
        elif type is 'horizzontal':
            return cv2.flip( frame, 0 )
        elif type is 'vertical':
            return cv2.flip( frame, 1 )
        elif type is 'both':
            return cv2.flip( frame, -1 )

    def extract_one_input(self, video_path, segment, pbar):
        one_input = np.zeros(shape=(config.frames_per_step, config.op_input_height, config.op_input_width, 7), dtype=float)
        extracted_frames = {}
        frame_list = []
        try:
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            linspace_frame = np.linspace(segment[0], segment[1], num=config.frames_per_step)
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
                    pbar.update(1)
                    z += 1
                except Exception as e:
                    # print('\n A Exception \n', e, frame)
                    pass
        except Exception as e:
            # print('\n B Exception\n', e, frame, im.shape, im_prev.shape)
            # print('\n B Exception\n', e)
            pass

        frame_list = extracted_frames.keys()
        extracted_frames = None
        return one_input, frame_list

    def trimmed_segment_extractor(self, trimmed, untrimmed):
        Bool = True
        while Bool:
            try:
                random.seed(time.time())
                if config.snow_ball:
                    entry_label = random.choice(list(self.snow_ball_labels_calculator()))
                else:
                    entry_label = random.choice(list(trimmed))
                entry_name = random.choice(trimmed[entry_label])
                path = entry_name['path']
                if path not in untrimmed.keys():
                    continue
                video = cv2.VideoCapture(path)
                fps = video.get(cv2.CAP_PROP_FPS)
                segment = entry_name['segment']
                if segment[1] == segment[0]:
                    continue
                min_end = (segment[0] + config.window_size)
                random_end = min_end + random.random() * (segment[1] - min_end)
                int_part = int(random_end / 1)
                decimal_part = round(float(segment[1] % 1 / config.window_size)) * config.window_size
                end_second = int_part + decimal_part
                end_frame = int(round(end_second * fps))
                start_frame = int(end_frame - config.window_size * fps + 1)
                if start_frame <= 1:
                    start_frame = 2
                label_count = 0
                tot_frames = end_frame - start_frame
                minimum_label = tot_frames * 0.5
                segment = [start_frame, end_frame]
                label_clip = {}
                count = 0
                for frame in range(start_frame, end_frame):
                    if frame not in untrimmed[path]:
                        print(frame_list)
                    label = untrimmed[path][frame]
                    if label not in label_clip:
                        label_clip[label] = 0
                    label_clip[label] += 1
                    count += 1
                    final_label = max(label_clip, key=label_clip.get)
                    if label_clip[final_label] > 0.5*count:
                        Bool = False
                        break
                    # else:
                        # print('\n', untrimmed[path], label, entry_label, [start_frame, end_frame])
            except Exception as e:
                # print('\ntrimmed_segment_extractor EXCEPTION\n', e)
                pass
        segment = [start_frame, end_frame]
        return segment, path

    def untrimmed_segment_extractor(self, untrimmed):
        path = random.choice(list(untrimmed.keys()))
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        max_frame = max(list(untrimmed[path].keys()))
        Bool = True
        while Bool:
            random_end = (config.window_size * fps) + random.random() * (float(max_frame / fps) - config.window_size)
            int_part = int(random_end / 1)
            decimal_part = round(float(random_end % 1 / config.window_size)) * config.window_size
            random_end = int_part + decimal_part
            end_frame = int(random_end * fps)
            if end_frame > max_frame:
                continue
            start_frame = int(end_frame - config.window_size * fps + 1)
            not_zero_count = 0
            for frame in range(start_frame, end_frame):
                # if frame not in untrimmed[path]:
                #     print(untrimmed[path].keys())
                #     print(frame)
                #     print(video.get(cv2.CAP_PROP_FRAME_COUNT))
                label = untrimmed[path][frame]
                if label is not 0:
                    not_zero_count += 1
                    if not_zero_count >= config.window_size * fps * 0:
                        Bool = False
                        break
        segment = [start_frame, end_frame]
        return segment, path

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
        final_label = max(label_clip, key=label_clip.get)
        return final_label

    def batch_generator(self, pbar, Train=True, Trimmed=False, Action=False):
        random.seed(time.time())
        segment_collection = []
        video_name_collection = []
        batch = np.zeros(shape=(config.Batch_size, config.frames_per_step, config.op_input_height, config.op_input_width, 7), dtype=float)
        labels = np.zeros(shape=(config.Batch_size, self.num_classes), dtype=int)
        next_labels = np.zeros(shape=(config.Batch_size, self.num_classes), dtype=int)
        c = np.zeros(shape=(config.Batch_size, config.hidden_states_dim), dtype=float)
        h = np.zeros(shape=(config.Batch_size, config.hidden_states_dim), dtype=float)

        # Selecting correct dataset
        if Train:
            if Action:
                trimmed_dataset = self.dataset.trimmed_train_dataset
            else:
                trimmed_dataset = self.dataset.trimmed_train_next
            untrimmed_dataset = self.dataset.untrimmed_train_dataset
            untrimmed_next = self.dataset.untrimmed_train_next
        else:
            if Action:
                trimmed_dataset = self.dataset.trimmed_val_dataset
            else:
                trimmed_dataset = self.dataset.trimmed_val_next
            untrimmed_dataset = self.dataset.untrimmed_val_dataset
            untrimmed_next = self.dataset.untrimmed_val_next

        j = 0
        while j < config.Batch_size:
            if Trimmed:
                segment, path = self.trimmed_segment_extractor(trimmed_dataset, untrimmed_dataset)
            else:
                segment, path = self.untrimmed_segment_extractor(untrimmed_dataset)

            one_input, frame_list = self.extract_one_input(path, segment, pbar)
            final_label = self.label_calculator(frame_list, path, untrimmed_dataset)
            next_final_label = self.label_calculator(frame_list, path, untrimmed_next)
            config.snow_ball_step_count += 1

            segment_collection.append(segment)
            video_name_collection.append(path)
            batch[j, :, :, :, :] = one_input
            labels[j, final_label] = 1
            next_labels[j, next_final_label] = 1
            if path in self.hidden_states_collection:
                start_frame = segment_collection[j][0] - 1
                if start_frame in self.hidden_states_collection[path].keys():
                    c[j, :] = self.hidden_states_collection[path][start_frame]['c']
                    h[j, :] = self.hidden_states_collection[path][start_frame]['h']

            j = j + 1

        pbar.update(1)
        pbar.refresh()
        return batch, labels, c, h, video_name_collection, segment_collection, next_labels

    def test_generator(self, pbar, path, segment):
        random.seed(time.time())
        video_name_collection = []
        batch = np.zeros(shape=(1, config.frames_per_step, config.op_input_height, config.op_input_width, 7), dtype=float)
        labels = np.zeros(shape=(1, self.num_classes), dtype=int)
        c = np.zeros(shape=(1, config.hidden_states_dim), dtype=float)
        h = np.zeros(shape=(1, config.hidden_states_dim), dtype=float)

        one_input, frame_list = self.extract_one_input(path, segment, pbar)
        final_label = self.label_calculator(frame_list, path, self.dataset.untrimmed_train_dataset)

        batch[0, :, :, :, :] = one_input
        label = self.dataset.id_to_label[final_label]

        if path in self.hidden_states_collection:
            start_frame = segment[0] - 1
            if start_frame in self.hidden_states_collection[path].keys():
                c[1, :] = self.hidden_states_collection[path][start_frame]['c']
                h[1, :] = self.hidden_states_collection[path][start_frame]['h']

        pbar.update(1)
        pbar.refresh()
        return batch, c, h, label

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
                    pafMat, heatMat = self.openpose.compute_pose_frame(X_data[i, j, :, :, :3])
                    norm_pafMat = pafMat
                    norm_pafMat = cv2.normalize(pafMat, norm_pafMat, 0, 255, cv2.NORM_MINMAX)
                    norm_heatMat = heatMat
                    norm_heatMat = cv2.normalize(heatMat, norm_heatMat, 0, 255, cv2.NORM_MINMAX)
                    X_data[i, j, :, :, 3] = heatMat
                    X_data[i, j, :, :, 4] = pafMat
                    final_frame = X_data[i, j, :, :, :]
                    shrinked_frame = cv2.resize(final_frame, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
                    shrinked_frame = self.augment_data(shrinked_frame, augment)
                    shrinked_X[i, j, :, :, :] = shrinked_frame
                    # # heatMat = cv2.resize(heatMat, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
                    # # pafMat = cv2.resize(pafMat, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
                    # im = X_data[i, j, :, :, :3]/255
                    # flow = X_data[i, j, :, :, 5:]
                    # # hsv = np.zeros((im.shape[0], im.shape[1], im.shape[2]))
                    # mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
                    # hsv = cv2.cvtColor(im.astype('uint8'), cv2.COLOR_RGB2HSV)
                    # hsv[:,:,0] = ang * (180/ np.pi / 2)
                    # hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                    # hsv = np.asarray(hsv, dtype= np.float32)
                    # rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                    # pafMat_grey = cv2.cvtColor(pafMat, cv2.COLOR_GRAY2BGR)
                    # heatMat_grey = cv2.cvtColor(heatMat, cv2.COLOR_GRAY2BGR)
                    # # print(pafMat_grey.shape, heatMat_grey.shape, shrinked_frame.shape)
                    # numpy_horizontal_concat = np.concatenate((im, heatMat_grey/255), axis=1)
                    # numpy_horizontal_concat_2 = np.concatenate((numpy_horizontal_concat, pafMat_grey/255), axis=1)
                    # numpy_horizontal_concat_3 = np.concatenate((numpy_horizontal_concat_2, rgb_flow), axis=1)
                    # cv2.imshow('ImageWindow',numpy_horizontal_concat_3)
                    # cv2.waitKey()
                    pbar.update(1)
            X[k]['X'] = shrinked_X
        pbar.refresh()
        pbar.clear()
        pbar.close()
        return X
