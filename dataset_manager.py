import json
import os
import cv2
from tqdm import tqdm
import random
import pprint
import time
import pickle
import config


pp = pprint.PrettyPrinter(indent=4)


class Dataset:
    def __init__(self):

        if os.path.isfile('dataset/label_to_id.pkl') and os.path.isfile('dataset/id_to_label.pkl'):
            self.label_to_id = self.load_dataset_obj('label_to_id')
            self.id_to_label = self.load_dataset_obj('id_to_label')
        else:
            self.label_to_id, self.id_to_label = self.create_labels_mappings()
            self.save_dataset_obj(self.label_to_id, 'label_to_id')
            self.save_dataset_obj(self.id_to_label, 'id_to_label')

        self.number_of_classes = len(self.id_to_label)
        pp.pprint(self.label_to_id)
        # pp.pprint(self.number_of_classes)

        if (os.path.isfile('dataset/trimmed_train_dataset.pkl') and
                os.path.isfile('dataset/trimmed_val_dataset.pkl') and
                os.path.isfile('dataset/untrimmed_train_dataset.pkl') and
                os.path.isfile('dataset/untrimmed_val_dataset.pkl') and
                os.path.isfile('dataset/untrimmed_train_next.pkl') and
                os.path.isfile('dataset/untrimmed_val_next.pkl')):

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
        self.validation_fraction = 0.02
        self.whole_dataset = self.load_origianl_dataset(whichDataset='Breakfast')
        self.Train_dataset, self.Val_dataset = self.split_dataset()
        pp.pprint(self.Val_dataset.keys())
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

    def load_origianl_dataset(self, whichDataset):
        if whichDataset is 'Ocado':
            json_data = open('dataset/temp.json').read()
            Dataset = json.loads(json_data)

            json_data_2 = open('dataset/temp2.json').read()
            Dataset_2 = json.loads(json_data_2)

            Dataset.update(Dataset_2)
        elif whichDataset is 'Breakfast':
            allow = ['coffee', 'cereals', 'milk'] #, 'scrambledegg', 'salat']
            json_data = open('dataset/Breakfast.json').read()
            Dataset = json.loads(json_data)
            new_Dataset = {}
            azioni = []
            for video in Dataset.keys():
                for annotation in Dataset[video]:
                    # print(annotation['activity'])
                    if annotation['activity'] in allow:
                        if annotation['label'] not in azioni:
                            azioni.append(annotation['label'])
                        if video not in new_Dataset:
                            new_Dataset[video]=[]
                        annotation['milliseconds'][0]=annotation['milliseconds'][0]*1000
                        annotation['milliseconds'][1]=annotation['milliseconds'][1]*1000
                        new_Dataset[video].append(annotation)
            Dataset = new_Dataset
            # pp.pprint(azioni)
        elif whichDataset is 'ActivityNet':
            json_data = open('dataset/activity_net.v1-3.min.json').read()
            original_dataset = json.loads(json_data)['database']
            Dataset = {}
            for video in original_dataset.keys():
                for annotation in original_dataset[video]['annotations']:
                    segment = annotation['segment']
                    if segment[1] > segment[0] and segment[1] <= original_dataset[video]['duration']:
                        new_name = video + '.mp4'
                        Dataset[new_name] = []
                        new_entry = {}
                        new_entry['label'] = annotation['label']
                        segment = annotation['segment']
                        millisecond_segment = [int(segment[0]) * 1000, int(segment[1]) * 1000]
                        new_entry['milliseconds'] = millisecond_segment
                        Dataset[new_name].append(new_entry)
        return Dataset

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
                if fl in dataset.keys() or fl.split('.')[0] in dataset.keys():
                    for annotation in dataset[fl]:
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
        # print('untrimmed dataset generation')
        iter_count = 0
        for root, dirs, files in os.walk('dataset'):
            for fl in files:
                if fl in dataset.keys():
                    iter_count += 1

        pbar = tqdm(total=(iter_count), leave=False, desc='Generating untrimmed dataset')

        whatLabel = 'label'
        if next:
            whatLabel = 'next_label'

        for root, dirs, files in os.walk('dataset'):
            for fl in files:
                if fl in dataset.keys():
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
                        for annotation in dataset[fl]:
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
