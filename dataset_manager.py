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

        self.whole_dataset = Annotation().Dataset

        if os.path.isfile('dataset/label_to_id.pkl') and os.path.isfile('dataset/id_to_label.pkl') and not config.rebuild:
            self.label_to_id = self.load_dataset_obj('label_to_id')
            self.id_to_label = self.load_dataset_obj('id_to_label')
        else:
            self.label_to_id, self.id_to_label = self.create_labels_mappings()
            self.save_dataset_obj(self.label_to_id, 'label_to_id')
            self.save_dataset_obj(self.id_to_label, 'id_to_label')
        self.number_of_classes = len(self.id_to_label)
        pp.pprint(self.id_to_label)

        if (os.path.isfile('dataset/trimmed_train_dataset.pkl') and
                os.path.isfile('dataset/trimmed_val_dataset.pkl') and
                os.path.isfile('dataset/untrimmed_train_dataset.pkl') and
                os.path.isfile('dataset/untrimmed_val_dataset.pkl') and
                os.path.isfile('dataset/untrimmed_train_next.pkl') and
                os.path.isfile('dataset/untrimmed_val_next.pkl') and
                not config.rebuild):

            print(not config.rebuild)
            self.trimmed_train_dataset = self.load_dataset_obj('trimmed_train_dataset')
            self.trimmed_val_dataset = self.load_dataset_obj('trimmed_val_dataset')
            self.trimmed_train_next = self.load_dataset_obj('trimmed_train_next')
            self.trimmed_val_next = self.load_dataset_obj('trimmed_val_next')
            self.untrimmed_train_dataset = self.load_dataset_obj('untrimmed_train_dataset')
            self.untrimmed_val_dataset = self.load_dataset_obj('untrimmed_val_dataset')
            self.untrimmed_train_next = self.load_dataset_obj('untrimmed_train_next')
            self.untrimmed_val_next = self.load_dataset_obj('untrimmed_val_next')
        else:
            self.generate_dataset()

    def generate_dataset(self):
        self.validation_fraction = config.validation_fraction
        self.Train_dataset, self.Val_dataset = self.split_dataset()
        self.untrimmed_train_dataset = self.create_untrimmed_collection(self.Train_dataset, next=False)
        self.untrimmed_val_dataset = self.create_untrimmed_collection(self.Val_dataset, next=False)
        self.untrimmed_train_next = self.create_untrimmed_collection(self.Train_dataset, next=True)
        self.untrimmed_val_next = self.create_untrimmed_collection(self.Val_dataset, next=True)
        self.trimmed_train_dataset = self.create_trimmed_collection(self.Train_dataset, next=False)
        self.trimmed_val_dataset = self.create_trimmed_collection(self.Val_dataset, next=False)
        self.trimmed_train_next = self.create_trimmed_collection(self.Train_dataset, next=True)
        self.trimmed_val_next = self.create_trimmed_collection(self.Val_dataset, next=True)
        self.save_dataset_obj(self.trimmed_train_dataset, 'trimmed_train_dataset')
        self.save_dataset_obj(self.trimmed_val_dataset, 'trimmed_val_dataset')
        self.save_dataset_obj(self.trimmed_train_next, 'trimmed_train_next')
        self.save_dataset_obj(self.trimmed_val_next, 'trimmed_val_next')
        self.save_dataset_obj(self.untrimmed_train_dataset, 'untrimmed_train_dataset')
        self.save_dataset_obj(self.untrimmed_val_dataset, 'untrimmed_val_dataset')
        self.save_dataset_obj(self.untrimmed_train_next, 'untrimmed_train_next')
        self.save_dataset_obj(self.untrimmed_val_next, 'untrimmed_val_next')

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

    def create_trimmed_collection(self, dataset, next):
        collection = {}
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
                    for annotation in dataset[key]:
                        if annotation[whatLabel] in self.label_to_id.keys():
                            label = annotation[whatLabel]
                            segment = annotation['milliseconds']
                            new_segment = [segment[0] / 1000, segment[1] / 1000]
                            if self.label_to_id[label] not in collection.keys():
                                collection[self.label_to_id[label]] = []
                            new_entry = {}
                            new_entry['path'] = path
                            new_entry['segment'] = new_segment
                            collection[self.label_to_id[label]].append(new_entry)
        return collection

    def create_untrimmed_collection(self, dataset, next):
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

    def save_dataset_obj(self, obj, name):
        with open('dataset/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_dataset_obj(self, name):
        with open('dataset/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
