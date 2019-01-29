import json
import os
import cv2
from tqdm import tqdm
import random
import pprint
import time
import pickle
import config
from annotation_generator import Annotation


pp = pprint.PrettyPrinter(indent=4)


class Dataset:
    def __init__(self):
        if (os.path.isfile('dataset/frame_now.pkl') and
                os.path.isfile('dataset/frame_next.pkl') and
                os.path.isfile('dataset/train_collection.pkl') and
                os.path.isfile('dataset/train_tree_list.pkl') and
                os.path.isfile('dataset/train_couple_count.pkl') and
                os.path.isfile('dataset/test_collection.pkl') and
                os.path.isfile('dataset/test_tree_list.pkl') and
                os.path.isfile('dataset/test_couple_count.pkl') and
                os.path.isfile('dataset/label_to_id.pkl') and
                os.path.isfile('dataset/id_to_label.pkl') and
                not config.rebuild):

            self.label_to_id = self.load('label_to_id')
            self.id_to_label = self.load('id_to_label')
            self.number_of_classes = len(self.id_to_label)
            self.frame_now = self.load('frame_now')
            self.frame_next = self.load('frame_next')
            self.train_collection = self.load('train_collection')
            self.train_tree_list = self.load('train_tree_list')
            self.train_couple_count = self.load('train_couple_count')
            self.test_collection = self.load('test_collection')
            self.test_tree_list = self.load('test_tree_list')
            self.test_couple_count = self.load('test_couple_count')
        else:
            self.generate_dataset()

        pp.pprint(self.id_to_label)


    def generate_dataset(self):
        self.whole_dataset = Annotation().Dataset
        self.label_to_id, self.id_to_label = self.create_labels_mappings()
        self.save(self.label_to_id, 'label_to_id')
        self.save(self.id_to_label, 'id_to_label')
        self.number_of_classes = len(self.id_to_label)
        self.validation_fraction = config.validation_fraction
        self.Train_dataset, self.Val_dataset = self.split_dataset()
        self.frame_now = self.compute_frame_label(self.whole_dataset, next=False)
        self.frame_next = self.compute_frame_label(self.whole_dataset, next=True)
        self.train_collection, self.train_tree_list, self.train_couple_count = self.new_collection(self.whole_dataset)
        self.test_collection, self.test_tree_list, self.test_couple_count = self.new_collection(self.whole_dataset)
        self.save(self.frame_now, 'frame_now')
        self.save(self.frame_next, 'frame_next')
        self.save(self.train_collection, 'train_collection')
        self.save(self.train_tree_list, 'train_tree_list')
        self.save(self.train_couple_count, 'train_couple_count')
        self.save(self.test_collection, 'test_collection')
        self.save(self.test_tree_list, 'test_tree_list')
        self.save(self.test_couple_count, 'test_couple_count')

    def split_dataset(self):
        dataset_train = self.whole_dataset
        validation = {}
        random.seed(time.time())
        entry_val = int(len(self.whole_dataset) * self.validation_fraction)
        for i in range(entry_val):
            entry_name = random.choice(list(dataset_train.keys()))
            validation[entry_name] = dataset_train[entry_name]
            self.whole_dataset.pop(entry_name)
        return dataset_train, validation

    def create_labels_mappings(self):
        label_to_id = {}
        id_to_label = {}
        label_to_id['NULL'] = 0
        id_to_label[0] = 'NULL'
        i = 1
        for video_name in self.whole_dataset:
            for segment in self.whole_dataset[video_name]:
                label = segment['label']
                if label == 'person:ask tool':
                    label = 'person:give tool'
                if label not in label_to_id:
                    label_to_id[label] = i
                    if label == 'person:give tool':
                        label_to_id['person:ask tool'] = i
                    id_to_label[i] = label
                    i += 1
        return label_to_id, id_to_label

    def compute_frame_label(self, dataset, next):
        collection = {}
        iter_count = 0
        for root, dirs, files in os.walk('dataset'):
            for fl in files:
                path = root + '/' + fl
                is_dataset = False
                if path in dataset.keys():
                    key = path
                    is_dataset = True
                elif fl in dataset.keys():
                    key = fl
                    is_dataset = True
                if is_dataset:
                    iter_count += 1

        pbar = tqdm(total=(iter_count), leave=False, desc='Generating untrimmed dataset')

        whatLabel = 'label'
        if next:
            whatLabel = 'next_label'

        for root, dirs, files in os.walk('dataset'):
            for fl in files:
                path = root + '/' + fl
                is_dataset = False
                if path in dataset.keys():
                    key = path
                    is_dataset = True
                elif fl in dataset.keys():
                    key = fl
                    is_dataset = True
                if is_dataset:
                    path = root + '/' + fl
                    video = cv2.VideoCapture(path)
                    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                    fps = video.get(cv2.CAP_PROP_FPS)
                    tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    if tot_frames is 0:
                        break
                    frames_label = {}
                    not_zero = False
                    for frame in range(tot_frames):
                        frame = frame + 1
                        frame_in_msec = (frame / float(fps)) * 1000
                        label = 'NULL'
                        for annotation in dataset[key]:
                            segment = annotation['milliseconds']
                            if frame_in_msec <= segment[1] and frame_in_msec >= segment[0]:
                                if annotation[whatLabel] in self.label_to_id.keys():
                                    label = annotation[whatLabel]
                                    not_zero = True
                                    break
                        frames_label[frame] = self.label_to_id[label]
                    collection[path] = {}
                    collection[path] = frames_label
                    pbar.update(1)
        pbar.close()
        return collection

    def new_collection(self, dataset):
        iter_count = 0
        for root, dirs, files in os.walk('dataset'):
            for fl in files:
                path = root + '/' + fl
                is_dataset = False
                if path in dataset.keys():
                    key = path
                    is_dataset = True
                elif fl in dataset.keys():
                    key = fl
                    is_dataset = True
                if is_dataset:
                    iter_count += 1

        pbar = tqdm(total=(iter_count), leave=False, desc='Generating untrimmed dataset')

        collection = {}
        couple_count =  {}
        tree_list = {}

        for root, dirs, files in os.walk('dataset'):
            for fl in files:
                path = root + '/' + fl
                is_dataset = False
                if path in dataset.keys():
                    key = path
                    is_dataset = True
                elif fl in dataset.keys():
                    key = fl
                    is_dataset = TrueFalse

                if is_dataset:
                    path = root + '/' + fl
                    video = cv2.VideoCapture(path)
                    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                    fps = video.get(cv2.CAP_PROP_FPS)
                    tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    tot_steps = int(tot_frames/(config.window_size*fps))
                    label_history = ''
                    if tot_steps is 0:
                        break
                    step_label = {}
                    not_zero = False
                    for step in range(tot_steps):
                        max_frame = int((step+1)*config.window_size*fps)+1
                        if max_frame > tot_frames:
                            continue
                        frame_list = [frame for frame in range(int(step*config.window_size*fps + 1),int((step+1)*config.window_size*fps)+1)]
                        segment = [frame_list[0], frame_list[-1]]
                        current_label = self.label_calculator(frame_list, path, self.frame_now)
                        next_label = self.label_calculator(frame_list, path, self.frame_next)
                        if len(label_history) is 0:
                            label_history += str(current_label)
                        elif str(current_label) not in label_history.split('-')[-1]:
                            label_history += '-' + str(current_label)

                        if label_history not in tree_list:
                            tree_list[label_history] = [next_label]
                        else:
                            if next_label not in tree_list[label_history]:
                                tree_list[label_history].append(next_label)

                        couple = str(current_label) + '-' + str(next_label)
                        if couple not in couple_count:
                            couple_count[couple] = 1
                        else:
                            couple_count[couple] += 1

                        entry = {'now_label' : current_label, 'next_label' : next_label, 'all_next_label' : couple,
                                 'path': path, 'segment':segment, 'history':label_history}
                        if couple not in couple_count:
                            collection[couple] = [entry]
                        else:
                            if couple not in collection:
                                collection[couple] = []
                            collection[couple].append(entry)

                    pbar.update(1)
        for x in collection:
            for entry in collection[x]:
                all_next_label = tree_list[entry['history']]
                entry['all_next_label'] = all_next_label
        pbar.close()
        return collection, tree_list, couple_count

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

    def save(self, obj, name):
        with open('dataset/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        with open('dataset/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
