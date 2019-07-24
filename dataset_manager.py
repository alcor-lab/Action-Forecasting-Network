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
import numpy as np
import copy
import csv


pp = pprint.PrettyPrinter(indent=4)


class Dataset:
    def __init__(self):
        if (os.path.isfile('dataset/frame_label.pkl') and
                os.path.isfile('dataset/train_collection.pkl') and
                os.path.isfile('dataset/test_collection.pkl') and
                os.path.isfile('dataset/ordered_collection.pkl') and
                os.path.isfile('dataset/label_to_id.pkl') and
                os.path.isfile('dataset/id_to_label.pkl') and
                os.path.isfile('dataset/word_to_id.pkl') and
                os.path.isfile('dataset/id_to_word.pkl') and
                os.path.isfile('dataset/now_weigth.pkl') and
                os.path.isfile('dataset/next_weigth.pkl') and
                os.path.isfile('dataset/help_weigth.pkl') and
                os.path.isfile('dataset/comb_count.csv') and
                not config.rebuild):

            self.label_to_id = self.load('label_to_id')
            self.id_to_label = self.load('id_to_label')
            self.word_to_id = self.load('word_to_id')
            self.id_to_word = self.load('id_to_word')
            self.number_of_classes = len(self.word_to_id)
            self.frame_now = self.load('frame_label')
            self.train_collection = self.load('train_collection')
            self.test_collection = self.load('test_collection')
            self.ordered_collection = self.load('ordered_collection')
            self.now_weigth = self.load('now_weigth')
            self.next_weigth = self.load('next_weigth')
            self.help_weigth = self.load('help_weigth')
        else:
            self.generate_dataset()

        pp.pprint(self.word_to_id)
        # pp.pprint(self.id_to_word)
        # pp.pprint(self.id_to_label)
        # pp.pprint(self.id_to_word)

    def generate_dataset(self):
        annotation = Annotation()
        self.whole_dataset = annotation.dataset
        self.label_to_id =  annotation.label_to_id
        self.id_to_label =  annotation.id_to_label
        self.frame_label =  annotation.frames_label
        self.object_label = annotation.object_label
        self.obj_id_2_label = annotation.obj_id_2_label
        self.location_label = annotation.location_label
        self.word_to_id, self.id_to_word = self.create_labels_mappings_network(self.label_to_id)
        self.number_of_classes = len(self.word_to_id)
        self.save(self.label_to_id, 'label_to_id')
        self.save(self.id_to_label, 'id_to_label')
        self.save(self.word_to_id, 'word_to_id')
        self.save(self.id_to_word, 'id_to_word')
        self.save(self.frame_label, 'frame_label')
        self.validation_fraction = config.validation_fraction
        self.collection, self.ordered_collection, self.multi_list, self.couple_count, self.max_history, self.comb_count= self.new_collection(self.whole_dataset)
        self.now_weigth, self.next_weigth, self.help_weigth = self.compute_weight(self.collection)
        non_zero_division = False
        while not non_zero_division:
            self.train_collection, self.test_collection = self.split_dataset_take(self.collection)
            non_zero_division = True
            for now in self.train_collection.keys():
                for next in self.train_collection[now].keys():
                    for help in self.train_collection[now][next].keys():
                        if len(self.train_collection[now][next][help]) == 0:
                            non_zero_division = False
        self.save(self.test_collection, 'test_collection')
        self.save(self.train_collection, 'train_collection')
        self.save(self.test_collection, 'test_collection')
        self.save(self.ordered_collection, 'ordered_collection')
        self.save(self.now_weigth, 'now_weigth')
        self.save(self.next_weigth, 'next_weigth')
        self.save(self.help_weigth, 'help_weigth')
        with open('dataset/comb_count.csv', 'w') as f:
            for key in self.comb_count.keys():
                f.write("%s,%s\n"%(key,self.comb_count[key]))

    def split_dataset_second(self, collection):
        dataset = copy.deepcopy(collection)
        validation = {}
        random.seed(time.time())
        entry_val = int(len(self.whole_dataset) * self.validation_fraction)
        i = 0
        while i < entry_val:
            if config.balance_key == 'all':
                r_now = random.choice(list(dataset.keys()))
                r_next = random.choice(list(dataset[r_now].keys()))
                r_help = random.choice(list(dataset[r_now][r_next].keys()))
                if len(dataset[r_now][r_next][r_help])< 2:
                    continue
                r_index = random.randrange(len(dataset[r_now][r_next][r_help]))
                entry = dataset[r_now][r_next][r_help][r_index]
                if r_now not in validation:
                    validation[r_now] = {}
                if r_next not in validation[r_now]:
                    validation[r_now][r_next] = {}
                if r_help not in validation[r_now][r_next]:
                    validation[r_now][r_next][r_help] = []
                validation[r_now][r_next][r_help].append(entry)
                del dataset[r_now][r_next][r_help][r_index]
            else:
                random_couple = random.choice(list(dataset))
                r_index = random.randrange(len(dataset[random_couple]))
                entry = dataset[random_couple][r_index]
                if random_couple not in validation:
                    validation[random_couple] = []
                validation[random_couple].append(entry)
                del dataset[random_couple][r_index]
            i = i + 1
        return dataset, validation

    def split_dataset_video(self, dataset):
        validation = {}
        train = {}
        random.seed(time.time())
        entry_val = int(len(self.ordered_collection) * self.validation_fraction)
        i = 0
        test_path = []
        while len(test_path) < entry_val + 1:
            new_test_video = random.choice(list(self.ordered_collection.keys()))
            if 'cam0' in new_test_video or new_test_video in test_path:
                continue
            test_path.append(new_test_video)
            if len(test_path) == entry_val + 1:
                test_with_robot = [x for x in test_path if 'robot' in x]
                if len(test_with_robot) <2:
                    test_path = []

        train_path = []
        for r_now in dataset.keys():
            for r_next in dataset[r_now].keys():
                for r_help in dataset[r_now][r_next].keys():
                    for entry in dataset[r_now][r_next][r_help]:
                        path = entry['path']
                        if path in test_path:
                            if r_now not in validation:
                                validation[r_now] = {}
                            if r_next not in validation[r_now]:
                                validation[r_now][r_next] = {}
                            if r_help not in validation[r_now][r_next]:
                                validation[r_now][r_next][r_help] = []
                            validation[r_now][r_next][r_help].append(entry)
                        else:
                            train_path.append(path)
                            if r_now not in train:
                                train[r_now] = {}
                            if r_next not in train[r_now]:
                                train[r_now][r_next] = {}
                            if r_help not in train[r_now][r_next]:
                                train[r_now][r_next][r_help] = []
                            train[r_now][r_next][r_help].append(entry)

        both_dataset = [x for x in test_path if x in train_path]
        # pp.pprint(both_dataset)
        # pp.pprint(train_path)
        test_path.sort()
        pp.pprint(test_path)
        # for j in test_path:
        #     if j in train_path:
        #         print(test_path)
                            
        return train, validation

    def split_dataset_take(self, dataset):
        validation = {}
        train = {}
        random.seed(time.time())
        take_collection = []
        for path in self.ordered_collection:
            take = path.split('/')[-1]
            take = take.split('_cam')[0]
            if take not in take_collection:
                take_collection.append(take)
        entry_val = int(len(take_collection) * self.validation_fraction)
        take_with_robot = [x for x in self.ordered_collection if 'robot' in x]
        take_with_robot = [x.split('/')[-1] for x in take_with_robot]
        take_with_robot = [x.split('_cam')[0] for x in take_with_robot]
        test_take = []
        while len(test_take) < entry_val + 1:
            new_test_take = random.choice(list(take_collection))
            test_take.append(new_test_take)
            if len(test_take) == entry_val + 1:
                test_with_robot = [x for x in test_take if x in take_with_robot]
                if len(test_with_robot) <1:
                    test_take = []

        test_path = []
        for path in self.ordered_collection:
            take = path.split('/')[-1]
            take = take.split('_cam')[0]
            if take in test_take:
                test_path.append(path)

        train_path = []
        for r_now in dataset.keys():
            for r_next in dataset[r_now].keys():
                for r_help in dataset[r_now][r_next].keys():
                    for entry in dataset[r_now][r_next][r_help]:
                        path = entry['path']
                        if path in test_path:
                            if r_now not in validation:
                                validation[r_now] = {}
                            if r_next not in validation[r_now]:
                                validation[r_now][r_next] = {}
                            if r_help not in validation[r_now][r_next]:
                                validation[r_now][r_next][r_help] = []
                            validation[r_now][r_next][r_help].append(entry)
                        else:
                            if path not in train_path:
                                train_path.append(path)
                            if r_now not in train:
                                train[r_now] = {}
                            if r_next not in train[r_now]:
                                train[r_now][r_next] = {}
                            if r_help not in train[r_now][r_next]:
                                train[r_now][r_next][r_help] = []
                            train[r_now][r_next][r_help].append(entry)

        both_dataset = [x for x in test_path if x in train_path]
        pp.pprint(both_dataset)
        pp.pprint(test_path)
        test_path.sort()
        pp.pprint(len(test_path))
        pp.pprint(len(train_path))
        # for j in test_path:
        #     if j in train_path:
        #         print(test_path)
                            
        return train, validation

    def create_labels_mappings_network(self, label_to_id):
        word_to_id = {}
        id_to_word = {}
        word_to_id['sil'] = 0
        id_to_word[0] = 'sil'
        word_to_id['go'] = 1
        id_to_word[1] = 'go'
        word_to_id['end'] = 2
        id_to_word[2] = 'end'
        obj_list = {'guard', 'cloth', 'torch', 'spray_bottle', 'table', 'pliers', 'screwdriver', 'brush', 'cutter', 'robot', 'ladder','closed_ladder','person'}
        i = 3
        for label in label_to_id.keys():
            label = label.split(' ')
            for word in label:
                if word not in word_to_id:
                    word_to_id[word] = i
                    id_to_word[i] = word
                    i += 1
        for obj in obj_list:
            if obj not in word_to_id:
                word_to_id[obj] = i
                id_to_word[i] = obj
                i += 1
        return word_to_id, id_to_word

    def compute_weight(self, collection):
        now_frequency_count = np.zeros(shape=(len(self.word_to_id)), dtype=np.float32)
        next_frequency_count = np.zeros(shape=(len(self.word_to_id)), dtype=np.float32)
        help_frequency_count = np.zeros(shape=(len(self.word_to_id)), dtype=np.float32)

        for current_label in collection.keys():
            for next_label in collection[current_label].keys():
                for help_label in collection[current_label][next_label].keys():
                    for entry in collection[current_label][next_label][help_label]:
                        current_word = self.word_to_id[self.id_to_label[entry['now_label']]]
                        next_word = self.word_to_id[self.id_to_label[entry['next_label']]]
                        help_word = self.id_to_label[entry['help']].split(' ')
                        for word_id in help_word:
                            new_word_id = self.word_to_id[word_id]
                            help_frequency_count[new_word_id] += 1
                        now_frequency_count[current_word] += 1
                        next_frequency_count[next_word] += 1
        now_frequency_count = (np.max(now_frequency_count) - now_frequency_count +1) / np.mean(now_frequency_count)
        next_frequency_count = (np.max(next_frequency_count) - next_frequency_count +1) / np.mean(next_frequency_count)
        help_frequency_count = (np.max(help_frequency_count) - help_frequency_count +1) / np.mean(help_frequency_count)

        return now_frequency_count, next_frequency_count, help_frequency_count

    def new_collection(self, dataset):
        collection = {}
        ordered_collection = {}
        couple_count =  {}
        tree_list = {}
        graph_list = {}
        video_by_history = {}
        files_path = {}
        comb_count = {}

        for root, dirs, files in os.walk(config.kit_path):
            for fl in files:
                path = root + '/' + fl
                if path in dataset.keys():
                    files_path[path] = path
                elif fl in dataset.keys():
                    files_path[fl] = path

        pbar = tqdm(total=(len(files_path)), leave=False, desc='Creating Annotation')
        max_history = 0
        for entry in files_path:
            path = files_path[entry]
            path = path.replace('\\', '/')
            video = cv2.VideoCapture(path)
            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            fps = video.get(cv2.CAP_PROP_FPS)
            tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            tot_steps = int(tot_frames/(config.window_size*fps))
            label_history = []
            step_history = []
            if tot_steps is 0:
                break
            for step in range(tot_steps):
                max_frame = int((step+1)*config.window_size*fps)+1
                if max_frame > tot_frames:
                    continue
                frame_list = [frame for frame in range(int(step*config.window_size*fps + 1),int((step+1)*config.window_size*fps)+1)]
                segment = [frame_list[0], frame_list[-1]]
                current_label = self.label_calculator(frame_list, path, 'now')
                next_label = self.label_calculator(frame_list, path, 'next')
                help_label = self.label_calculator(frame_list, path, 'help')
                mask_frame_list =  [max(f-fps, 1) for f in frame_list]
                obj_label = self.object_return(mask_frame_list, path)
                location_label = self.location_return(mask_frame_list, path)
                # if current_label == 0:
                    # continue
                if len(label_history) == 0:
                    label_history.append(current_label)
                elif current_label != label_history[-1]:
                    label_history.append(current_label)


                if len(label_history) > max_history:
                    max_history = len(label_history)

                if tuple(label_history) not in tree_list:
                    tree_list[tuple(label_history)] = [next_label]
                else:
                    if next_label not in tree_list[tuple(label_history)]:
                        tree_list[tuple(label_history)].append(next_label)

                if current_label not in graph_list:
                    graph_list[current_label] = [next_label]
                else:
                    if next_label not in graph_list[current_label]:
                        graph_list[current_label].append(next_label)
                
                step_history.append(current_label)
                comb = ''
                if step > 2:
                    for step_label in step_history[-4:]:
                        comb +=self.id_to_label[step_label] + '-'
                    comb += self.id_to_label[next_label] + '-' + self.id_to_label[help_label]
                    comb = comb.replace(' ', '-')
                    if comb not in comb_count:
                        comb_count[comb] = 1
                    else:
                        comb_count[comb] += 1


                couple = str(current_label) + '-' + str(next_label)
                if couple not in couple_count:
                    couple_count[couple] = 1
                else:
                    couple_count[couple] += 1


                entry = {'now_label' : current_label, 'next_label' : next_label, 'all_next_label' : couple,
                         'path': path, 'segment':segment, 'history':label_history, 'time_step': step,
                         'help': help_label, 'step_history': step_history, 'obj_label': obj_label, 'location_label': location_label}
                if path not in ordered_collection:
                    ordered_collection[path] = {}
                ordered_collection[path][step] = entry
                 
                if config.balance_key != 'all':
                    if config.balance_key is 'now':
                        balance = current_label
                    elif config.balance_key is 'next':
                        balance = next_label
                    elif config.balance_key is 'couple':
                        balance = couple
                    if balance not in collection:
                        collection[balance] = [entry]
                    else:
                        collection[balance].append(entry)
                elif config.balance_key == 'all':
                    if current_label not in collection:
                        collection[current_label] = {}
                    if next_label not in collection[current_label]:
                        collection[current_label][next_label] = {}
                    if help_label not in collection[current_label][next_label]:
                        collection[current_label][next_label][help_label] = []
                    collection[current_label][next_label][help_label].append(entry)

            pbar.update(1)
        for x in collection:
            if config.balance_key == 'all':
                for now_label in collection[x]:
                    for next_label in collection[x][now_label]:
                        for entry in collection[x][now_label][next_label]:
                            if config.tree_or_graph is 'tree':
                                all_next_label = tree_list[tuple(entry['history'])]
                            elif config.tree_or_graph is 'graph':
                                all_next_label = graph_list[entry['now_label']]
                            entry['all_next_label'] = all_next_label
            else:
                for entry in collection[x]:
                    if config.tree_or_graph is 'tree':
                        all_next_label = tree_list[entry['history']]
                    elif config.tree_or_graph is 'graph':
                        all_next_label = graph_list[entry['now_label']]
                    entry['all_next_label'] = all_next_label
        if config.tree_or_graph is 'tree':
            multi_list = tree_list
        elif config.tree_or_graph is 'graph':
            multi_list = graph_list
        transition = np.zeros(shape=(len(self.id_to_label), len(self.id_to_label)), dtype=float)
        total = 0
        for x in couple_count:
            now = int(x.split('-')[0])
            next = int(x.split('-')[1])
            transition[now,next] += couple_count[x]
        for i in range(transition.shape[0]):
            tot_row = 0
            for j in range(transition.shape[1]):
                tot_row +=  transition[i,j]
            for j in range(transition.shape[1]):
                transition[i,j] /=  tot_row

        pbar.close()
        return collection, ordered_collection, multi_list, couple_count, max_history, comb_count

    def label_calculator(self, frame_list, path, next_current):
        label_clip = {}
        for frame in frame_list:
            if frame not in self.frame_label[path]:
                print(frame_list)
                print(frame_list)
            label = self.frame_label[path][frame][next_current]
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

    def object_return(self, frame_list, path):
        cut_name = path.split('/')[-1]
        cut_name = cut_name.split('cam')[0]
        frame = str(int((frame_list[0]+frame_list[-1])/2))
        out = {}
        if cut_name in self.object_label:
            obj_list = self.object_label[cut_name][frame][0]
            for idx in range(len(obj_list['object_list'])):
                obj_id = obj_list['object_list'][idx]
                obj_prob = obj_list['scores'][idx]
                obj_word = self.obj_id_2_label[obj_id]
                out[obj_word] = obj_prob
        return out

    def location_return(self, frame_list, path):
        cut_name = path.split('/')[-1]
        cut_name = cut_name.split('cam')[0]
        frame = str(int((frame_list[0]+frame_list[-1])/2))
        out = {}
        if cut_name in self.location_label:
            location = self.location_label[cut_name][frame]
            if type(location) is str:
                if location != 'No person' and location !='No diverter':
                    if location == "Technician under Diverter":
                        location = 'under_diverter'
                    if location == "Technician on the Ladder":
                        location = 'on_ladder'
                    if location == "Technician next to Guard":
                        location = 'at_guard_support'
                    out[location] = 1.0
        return out

    def save(self, obj, name):
        with open('dataset/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        with open('dataset/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

# Dataset_manager = Dataset()