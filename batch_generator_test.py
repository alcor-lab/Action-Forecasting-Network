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
from PIL import Image
import datetime
import prep_dataset_manager as prep_dataset


pp = pprint.PrettyPrinter(indent=4)

class IO_manager:
    def __init__(self, sess):
        self.dataset = Dataset()
        self.prep_dataset = prep_dataset.prep_dataset()
        self.num_classes = self.dataset.number_of_classes
        self.sess = sess
        self.no_sil_count = 0
        self.openpose = None
        self.snow_ball_minimum = config.snow_ball_classes
        self.chosen_label = {'now':{}, 'next':{}, 'activity':{}}
        if os.path.isfile('dataset/hidden_states_collection.pkl') and config.reuse_HSM:
            with open('dataset/hidden_states_collection.pkl', 'rb') as f:
                self.hidden_states_collection = pickle.load(f)
        else:
            self.hidden_states_collection = {}

        if os.path.isfile('dataset/output_collection.pkl') and config.reuse_output_collection:
            with open('dataset/output_collection.pkl', 'rb') as f:
                self.output_collection = pickle.load(f)
        else:
            self.output_collection = {}

        self.hidden_states_statistics()

    def compute_batch(self, pbar, Devices, Train, augment=True):
        def multiprocess_batch(x):
            X, Y, c, h, video_name_collection, segment_collection, next_label, help_label, now_weight, next_weight, help_weight, obj_input= self.batch_generator(pbar, Devices, Train)
            return {'X': X, 'Y': Y, 'c': c, 'h': h,
                    'next_Y': next_label,
                    'help_Y': help_label,
                    'video_name_collection': video_name_collection,
                    'segment_collection': segment_collection,
                    'now_weight': now_weight,
                    'next_weight': next_weight,
                    'help_weight': help_weight,
                    'obj_input': obj_input}

        pool = mp.Pool(processes=config.processes)
        if not Train:
            augment = False

        if Train:
            ready_batch = pool.map(multiprocess_batch, range(0, config.tasks))
        else:
            ready_batch = pool.map(multiprocess_batch, range(0, config.val_task))

        if not config.use_prep:
            ready_batch = self.add_pose(ready_batch, self.sess, augment)
        else:
            ready_batch = self.augment_input(ready_batch, self.sess, augment)

        # ready_batch = self.group_batches(ready_batch, Devices, pbar)
        pbar.close()
        pool.close()
        pool.join()
        return ready_batch

    def batch_generator(self, pbar, Devices, Train=True):
        random.seed(time.time())
        batch_segment_collection = []
        batch_video_name_collection = []
        if config.use_prep:
            batch = np.zeros(shape=(Devices, config.Batch_size, config.seq_len, config.frames_per_step, config.out_H, config.out_W, 7), dtype=np.uint8)
        else:
            batch = np.zeros(shape=(Devices, config.Batch_size, config.seq_len, config.frames_per_step, config.op_input_height, config.op_input_width, 7), dtype=np.uint8)
        labels = np.zeros(shape=(Devices, config.Batch_size,config.seq_len + 1), dtype=int)
        help_labels = np.zeros(shape=(Devices, config.Batch_size, 4), dtype=int)
        next_labels = np.zeros(shape=(Devices, config.Batch_size), dtype=int)
        c = np.zeros(shape=(Devices, len(config.encoder_lstm_layers), config.Batch_size, config.hidden_states_dim), dtype=float)
        h = np.zeros(shape=(Devices, len(config.encoder_lstm_layers), config.Batch_size, config.hidden_states_dim), dtype=float)
        now_weight = np.zeros(shape=(Devices, config.Batch_size, config.seq_len + 1), dtype=float)
        next_weight = np.zeros(shape=(Devices, config.Batch_size), dtype=float)
        help_weight = np.zeros(shape=(Devices, config.Batch_size, 4), dtype=float)
        obj_input = np.zeros(shape=(Devices, config.Batch_size,config.seq_len, len(self.dataset.word_to_id)), dtype=float)

        # Selecting correct dataset
        if Train:
            collection_dataset = self.dataset.train_collection
        else:
            collection_dataset = self.dataset.test_collection
        ordered_collection = self.dataset.ordered_collection

        d = 0
        while d < Devices:
            segment_collection = []
            video_name_collection = []
            j = 0
            while j < config.Batch_size:
                entry_list = self.entry_selector(collection_dataset,ordered_collection, config.is_ordered)
                s = 0
                for entry in entry_list:
                    current_label = entry['now_label']
                    current_label = self.dataset.word_to_id[self.dataset.id_to_label[current_label]]
                    next_label = entry['next_label']
                    next_label = self.dataset.word_to_id[self.dataset.id_to_label[next_label]]
                    path = entry['path']
                    segment = entry['segment']
                    help_label_raw = entry['help']
                    help_label = self.dataset.id_to_label[help_label_raw].split(' ')
                    if config.use_prep:
                        one_input, frame_list = self.extract_preprocessed_one_input(path, segment, pbar)
                    else:
                        one_input, frame_list = self.extract_one_input(path, segment, pbar)
                    config.snow_ball_step_count += 1

                    
                    batch[d, j, s, :, :, :, :] = one_input
                    labels[d, j, s] = current_label
                    now_weight[d, j, s] = self.dataset.now_weigth[current_label]

                    obj_label = entry['obj_label']
                    for obj in obj_label.keys():
                        position = self.dataset.word_to_id[obj]
                        value = obj_label[obj]
                        obj_input[d, j, s, position] = value

                    if config.add_location:
                        location_label = entry['location_label']
                        for loc in location_label.keys():
                            position = self.dataset.word_to_id[loc]
                            value = location_label[loc]
                            obj_input[d, j, s, position] = value

                    if s == 0:
                        if path in self.hidden_states_collection:
                            start_frame = segment[0] - 1
                            if start_frame in self.hidden_states_collection[path].keys():
                                c[d, :, j, :] = self.hidden_states_collection[path][start_frame]['c']
                                h[d, :, j, :] = self.hidden_states_collection[path][start_frame]['h']

                    if s == config.seq_len - 1:
                        segment_collection.append(segment)
                        video_name_collection.append(path)
                        next_labels[d, j] = next_label
                        next_weight[d, j] = self.dataset.next_weigth[next_label]
                        for n in range(3):
                            if n > len(help_label) - 1:
                                help_label_step = self.dataset.word_to_id['sil']
                            else:
                                help_label_step = self.dataset.word_to_id[help_label[n]]
                            help_labels[d, j, n] = help_label_step
                            help_weight[d, j, n] = self.dataset.help_weigth[help_label_step]
                        help_labels[d, j, 3] = self.dataset.word_to_id['end']
                        

                    s = s + 1
                labels[d, j, s] = self.dataset.word_to_id['end']
                j = j + 1
                if self.no_sil_count < config.no_sil_step:
                    self.no_sil_count += 1

            batch_video_name_collection.append(video_name_collection)
            batch_segment_collection.append(segment_collection)
            d = d + 1

        # pp.pprint(history)
        pbar.update(1)
        pbar.refresh()
        return batch, labels, c, h, batch_video_name_collection, batch_segment_collection, next_labels, help_labels, now_weight, next_weight, help_weight, obj_input

    def entry_selector(self, dataset,ordered_collection, is_ordered):
        random.seed(time.time())
        time_step = 0
        # start from first seq_len segment : to be improved
        while time_step <= config.seq_len:
            if config.balance_key == 'all':
                r_now = random.choice(list(dataset.keys()))
                if r_now == 0 and self.no_sil_count < config.no_sil_step:
                    continue
                r_next = random.choice(list(dataset[r_now].keys()))
                r_help = random.choice(list(dataset[r_now][r_next].keys()))
                try:
                    entry = random.choice(dataset[r_now][r_next][r_help])
                except Exception as e:
                    print(self.dataset.id_to_label[r_now], self.dataset.id_to_label[r_next], self.dataset.id_to_label[r_help])
                    print(r_now, r_next, r_help)
                    print(dataset[r_now][r_next][r_help])
                    pass
                time_step = entry['time_step']
            else:
                random_couple = random.choice(list(dataset))
                entry = random.choice(dataset[random_couple])

        entry_list = []
        back = False
        for _ in range(config.seq_len):
                if not back:
                    entry_list.append(entry)
                    back = True
                else:
                    path = entry['path']
                    time_step = entry['time_step']
                    entry = ordered_collection[path][time_step - 1]
                    entry_list.append(entry)

        entry_list.reverse()
        return entry_list

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
    def extract_preprocessed_one_input(self, video_path, segment, pbar):
        one_input = np.zeros(shape=(config.frames_per_step, config.out_H, config.out_W, 7), dtype=np.uint8)
        extracted_frames = {}
        frame_list = []
        try:
            linspace_frame = np.linspace(segment[0], segment[1], num=config.frames_per_step)
            z = 0
            for frame in linspace_frame:
                try:
                    one_input[z, :, :, :] = self.prep_dataset.get_matrix(video_path, frame)
                    z += 1
                except Exception as e:
                    print(e)
                    pass
                pbar.update(1)
        except Exception as e:
            print(e)
            pass
        frame_list = extracted_frames.keys()
        return one_input, frame_list

    def extract_one_input(self, video_path, segment, pbar):
        one_input = np.zeros(shape=(config.frames_per_step, config.op_input_height, config.op_input_width, 7), dtype=np.uint8)
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
                    norm_flow = norm_flow.astype(np.uint8)
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
        total = len(X) * shape[0] * shape[1]* shape[2]* shape[2]
        augment = 'none'
        if augment:
            augment = random.choice(['none', 'horizzontal', 'vertical', 'both'])
        pbar = tqdm(total=total, leave=False, desc='Computing Poses')
        for k in range(len(X)):
            X_data = (X[k])['X']
            shape = X_data.shape
            shrinked_X = np.zeros(shape=(shape[0], shape[1], shape[2], shape[3], config.out_H, config.out_W, shape[6]), dtype=np.uint8)
            for d in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        for z in range(shape[3]):
                            try:
                                X_Data = X_data[d, i, j, z, :, :, :3].astype(np.uint8)
                                pafMat, heatMat = self.openpose.compute_pose_frame(X_Data)
                                heatMat = heatMat.astype(np.uint8)
                                pafMat = pafMat.astype(np.uint8)
                                X_data[d, i, j, z, :, :, 3] = heatMat
                                X_data[d, i, j, z, :, :, 4] = pafMat
                                final_frame = X_data[d, i, j, z, :, :, :]
                                shrinked_frame = cv2.resize(final_frame, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
                                shrinked_frame = self.augment_data(shrinked_frame, augment)
                                shrinked_X[d, i, j, z, :, :, :] = shrinked_frame
                                pbar.update(1)
                            except Exception as e:
                                print(e)
                                pass
            if config.show_pic:
                self.show_input_pic(X_data)
            X[k]['X'] = shrinked_X
        pbar.refresh()
        pbar.clear()
        pbar.close()
        return X

    def augment_input(self, X, sess, augment=True):
        shape = (X[0])['X'].shape
        total = len(X) * shape[0] * shape[1]* shape[2]* shape[2]
        augment = 'none'
        if augment:
            augment = random.choice(['none', 'horizzontal', 'vertical', 'both'])
        pbar = tqdm(total=total, leave=False, desc='Computing Poses')
        for k in range(len(X)):
            X_data = (X[k])['X']
            shape = X_data.shape
            for d in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        for z in range(shape[3]):
                            Frame_Data = X_data[d, i, j, z, :, :, :]
                            augmented = self.augment_data(Frame_Data, augment)
                            X_data[d, i, j, z, :, :, :] = augmented
                            pbar.update(1)
            X[k]['X'] = X_data
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
                im = np.asarray(im, dtype= np.float32)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                hsv[:,:,0] = ang * (180/ np.pi / 2)
                hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                hsv = np.asarray(hsv, dtype= np.float32)
                heatMat = np.asarray(heatMat, dtype= np.float32)
                pafMat = np.asarray(pafMat, dtype= np.float32)
                rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                pafMat_grey = cv2.cvtColor(pafMat, cv2.COLOR_GRAY2BGR)
                heatMat_grey = cv2.cvtColor(heatMat, cv2.COLOR_GRAY2BGR)
                # print(im.shape,pafMat_grey.shape,heatMat_grey.shape,rgb_flow.shape)
                numpy_horizontal_concat = np.concatenate((im, pafMat_grey/255), axis=1)
                numpy_horizontal_concat_2 = np.concatenate((numpy_horizontal_concat, heatMat_grey/255), axis=1)
                numpy_horizontal_concat_3 = np.concatenate((numpy_horizontal_concat_2, rgb_flow), axis=1)
                # cv2.imshow('ImageWindow',numpy_horizontal_concat_3)
                # cv2.waitKey()
                im = Image.fromarray(np.uint8(numpy_horizontal_concat_2*255))
                im.save('picshow/' + str(datetime.datetime.now())+".jpeg")

    def save_hidden_state_collection(self):
        with open('dataset/hidden_states_collection.pkl', 'wb') as f:
            pickle.dump(self.hidden_states_collection, f, pickle.HIGHEST_PROTOCOL)

    def save_output_collection(self):
        with open('dataset/output_states_collection.pkl', 'wb') as f:
            pickle.dump(self.output_collection, f, pickle.HIGHEST_PROTOCOL)

    def add_hidden_state(self, video_name, frame, h, c):
        if video_name not in self.hidden_states_collection:
            self.hidden_states_collection[video_name] = {}
        if frame not in self.hidden_states_collection[video_name]:
                self.hidden_states_collection[video_name][frame] = {}
                self.hidden_states_collection[video_name][frame]['number_of_visits'] = 0
        self.hidden_states_collection[video_name][frame]['h'] = h
        self.hidden_states_collection[video_name][frame]['c'] = c
        self.hidden_states_collection[video_name][frame]['number_of_visits'] += 1
        
    def add_output_collection(self, video_name, segment, now_label, now_softmax, next_label, next_softmax, help_label, help_softmax):
        if video_name not in self.output_collection:
            self.output_collection[video_name] = {}
        if segment not in self.output_collection[video_name]:
                self.output_collection[video_name][segment] = {}
        self.output_collection[video_name][segment]['now_label'] = now_label
        self.output_collection[video_name][segment]['now_softmax'] = now_softmax
        self.output_collection[video_name][segment]['next_label'] = next_label
        self.output_collection[video_name][segment]['next_softmax'] = next_softmax
        self.output_collection[video_name][segment]['help_label'] = help_label
        self.output_collection[video_name][segment]['help_softmax'] = help_softmax

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
