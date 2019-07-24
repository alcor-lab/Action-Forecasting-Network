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
from prep_dataset_manager import prep_dataset


pp = pprint.PrettyPrinter(indent=4)

class IO_manager:
    def __init__(self, sess):
        self.dataset = Dataset()
        self.prep_dataset = prep_dataset()
        self.num_classes = self.dataset.number_of_classes
        self.num_activities = self.dataset.number_of_activities
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
            X, Y, c, h, video_name_collection, segment_collection, next_label= self.batch_generator(pbar, Train)
            return {'X': X, 'Y': Y, 'c': c, 'h': h,
                    'next_Y': next_label,
                    'video_name_collection': video_name_collection,
                    'segment_collection': segment_collection}

        pool = mp.Pool(processes=config.processes)
        ready_batch = pool.map(multiprocess_batch, range(0, config.tasks*Devices))
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
            video_name_collection = []
            segment_collection = []
            for i in range(Devices):
                video_name_collection.append(ready_batch[j+i]['video_name_collection'])
                segment_collection.append(ready_batch[j+i]['segment_collection'])
            new_collection.append({'X': X, 'Y': Y, 'c': c, 'h': h,
                    'next_Y': next_label,
                    'video_name_collection': video_name_collection,
                    'segment_collection': segment_collection})
            pbar.update(1)
        return new_collection

    def batch_generator(self, pbar, Train=True):
        random.seed(time.time())
        segment_collection = []
        video_name_collection = []
        batch = np.zeros(shape=(config.Batch_size, config.seq_len, config.frames_per_step, config.op_input_height, config.op_input_width, 7), dtype=np.uint8)
        labels = np.zeros(shape=(config.Batch_size,config.seq_len + 1), dtype=int)
        next_labels = np.zeros(shape=(config.Batch_size), dtype=int)
        c = np.zeros(shape=(len(config.encoder_lstm_layers), config.Batch_size, config.hidden_states_dim), dtype=float)
        h = np.zeros(shape=(len(config.encoder_lstm_layers), config.Batch_size, config.hidden_states_dim), dtype=float)

        # Selecting correct dataset
        if Train:
            dataset = self.dataset.train_collection
        else:
            dataset = self.dataset.test_collection
        ordered_collection = self.dataset.ordered_collection

        j = 0
        while j < config.Batch_size:
            entry_list = self.entry_selector(dataset,ordered_collection, config.is_ordered)
            s = 0
            for entry in entry_list:
                current_label = entry['now_label']
                next_label = entry['next_label']
                path = entry['path']
                segment = entry['segment']
                one_input, frame_list = self.extract_one_input(path, segment, pbar)
                config.snow_ball_step_count += 1

                
                batch[j, s, :, :, :, :] = one_input
                labels[j, s] = current_label
                next_labels[j] = next_label

                if s == 0:
                    if path in self.hidden_states_collection:
                        start_frame = segment[0] - 1
                        if start_frame in self.hidden_states_collection[path].keys():
                            c[j, :] = self.hidden_states_collection[path][start_frame]['c']
                            h[j, :] = self.hidden_states_collection[path][start_frame]['h']

                if s == config.seq_len - 1:
                    segment_collection.append(segment)
                    video_name_collection.append(path)

                s = s + 1
            labels[j, s] = self.dataset.label_to_id('end')
            j = j + 1

        # pp.pprint(history)
        pbar.update(1)
        pbar.refresh()
        return batch, labels, c, h, video_name_collection, segment_collection, next_labels,

    def entry_selector(self, dataset,ordered_collection, is_ordered):
        random.seed(time.time())
        time_step = 0
        # start from first seq_len segment : to be improved
        while time_step <= config.seq_len:
            if config.balance_key == 'all':
                r_activity = random.choice(list(dataset.keys()))
                r_now = random.choice(list(dataset[r_activity].keys()))
                r_next = random.choice(list(dataset[r_activity][r_now].keys()))
                entry = random.choice(dataset[r_activity][r_now][r_next])
                time_step = entry['time_step']

                if r_activity not in self.chosen_label['activity']:
                    self.chosen_label['activity'][r_activity] = 1
                else:
                    self.chosen_label['activity'][r_activity] += 1

                if r_next not in self.chosen_label['next']:
                    self.chosen_label['next'][r_next] = 1
                else:
                    self.chosen_label['next'][r_next] += 1

                if r_now not in self.chosen_label['now']:
                    self.chosen_label['now'][r_now] = 1
                else:
                    self.chosen_label['now'][r_now] += 1
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

    def extract_one_input(self, video_path, segment, pbar):
        one_input = np.zeros(shape=(config.frames_per_step, config.op_input_height, config.op_input_width, 7), dtype=float)
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
                    pass
                pbar.update(1)
        except Exception as e:
            pass
        frame_list = extracted_frames.keys()
        return one_input, frame_list

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

    def add_output_collection(self, video_name, frame, multi, multi_target, forecast, forecast_target):
        if video_name not in self.output_collection:
            self.output_collection[video_name] = {}
        if frame not in self.output_collection[video_name]:
                self.output_collection[video_name][frame] = {}
                self.output_collection[video_name][frame]['number_of_visits'] = 0
        self.output_collection[video_name][frame]['multi'] = multi
        self.output_collection[video_name][frame]['multi_target'] = multi_target
        self.output_collection[video_name][frame]['forecast'] = forecast
        self.output_collection[video_name][frame]['forecast_target'] = forecast_target
        self.output_collection[video_name][frame]['number_of_visits'] += 1

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

    def augment_data(self, frame, type):
        if type is 'none':
            return frame
        elif type is 'horizzontal':
            return cv2.flip( frame, 0 )
        elif type is 'vertical':
            return cv2.flip( frame, 1 )
        elif type is 'both':
            return cv2.flip( frame, -1 )
