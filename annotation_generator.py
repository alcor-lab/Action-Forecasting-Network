import json
import config
import os
import pprint
from tqdm import tqdm
import cv2
pp = pprint.PrettyPrinter(indent=4)

class Annotation:
    def __init__(self):
        json_data = open(config.kit_activity_annotation).read()
        activity_dataset = json.loads(json_data)
        json_data = open(config.kit_help_annotation).read()
        help_dataset = json.loads(json_data)
        json_data = open(config.kit_obj_annotation).read()
        self.object_label = json.loads(json_data)
        json_data = open(config.kit_loc_annotation).read()
        self.location_label = json.loads(json_data)
        self.dataset, self.frames_label, self.label_to_id, self.id_to_label = self.create_ocado_annotation(activity_dataset, help_dataset)

        self.obj_id_2_label = { 0: 'BG',
                                1: 'spray_bottle',
                                2: 'screwdriver',
                                3: 'torch',
                                4: 'cloth',
                                5: 'cutter',
                                6: 'pliers',
                                7: 'brush',
                                8: 'torch_handle',
                                9: 'guard',
                                10: 'ladder',
                                11: 'closed_ladder',
                                12: 'person',
                                13: 'table'}

        # pp.pprint(self.dataset)
        # pp.pprint(self.frames_label)
        # pp.pprint(self.label_to_id)
        # pp.pprint(self.id_to_label)
        # pp.pprint(self.object_label.keys())
            
    def create_ocado_annotation(self, activity_dataset, help_dataset):
        label_collection = []
        all_file = []
        all_path = []
        file_to_path = {}
        for root, dirs, files in os.walk(config.kit_path):
            all_file += files
            for fl in files:
                path = root + '/' + fl
                path= path.replace('\\', '/')
                all_path.append(path)
                file_to_path[fl] = path
        
        addition_activity = {}
        for video_name in activity_dataset:
            base_name = video_name.split('_cam')[0]
            entry = activity_dataset[video_name]
            for fl in all_file:
                if fl not in activity_dataset:
                    fl_base_name = fl.split('_cam')[0]
                    if fl_base_name == base_name and video_name != fl:
                        addition_activity[fl] = entry

        addition_help = {}
        for video_name in help_dataset:
            base_name = video_name.split('_cam')[0]
            entry = help_dataset[video_name]
            for fl in all_file:
                if fl not in help_dataset:
                    fl_base_name = fl.split('_cam')[0]
                    if fl_base_name == base_name and video_name != fl:
                        addition_help[fl] = entry

        help_dataset.update(addition_help)
        activity_dataset.update(addition_activity)

        del_activity = []
        del_help = []
        for fl in activity_dataset:
            if fl not in all_file:
                del_activity.append(fl)
                if fl in help_dataset:
                    del_help.append(fl)
            elif fl not in help_dataset:
                del_activity.append(fl)
                
        for fl in help_dataset:
            if fl not in all_file:
                del_help.append(fl)
                if fl in activity_dataset:
                    del_activity.append(fl)
            elif fl not in activity_dataset:
                del_help.append(fl)

        for fl in list(dict.fromkeys(del_activity)):
            del activity_dataset[fl]
        for fl in list(dict.fromkeys(del_help)):
            del help_dataset[fl]

        for fl in activity_dataset:
            if fl in activity_dataset:
                path = file_to_path[fl]
                video = cv2.VideoCapture(path)
                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 2)
                fps = video.get(cv2.CAP_PROP_FPS)

                for index in range(len(activity_dataset[fl])):
                    label = activity_dataset[fl][index]['label']
                    label = self.clean_label(label, id = 1)
                    activity_dataset[fl][index]['label'] = label
                    if label not in label_collection:
                        label_collection.append(label)

                for index in range(len(help_dataset[fl])):
                    label = help_dataset[fl][index]['label']
                    label = self.clean_label(label, id = 0)
                    help_dataset[fl][index]['label'] = label
                    if label not in label_collection:
                        label_collection.append(label)

        label_to_id, id_to_label = self.create_labels_mappings(label_collection)
        frames_label = self.compute_frame_label(activity_dataset, help_dataset, config.kit_path, label_to_id)

        for fl in activity_dataset:
            path = file_to_path[fl]
            video = cv2.VideoCapture(path)
            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 2)
            fps = video.get(cv2.CAP_PROP_FPS)
            for index in range(len(activity_dataset[fl])):
                segment = activity_dataset[fl][index]['milliseconds']
                # label = activity_dataset[fl][index]['label']
                # segment_len = float((segment[1] - segment[0]))/1000
                # print(label, segment_len)
                frame_start = int((segment[0]*fps)/1000)
                frame_end = int((segment[1]*fps)/1000)
                next_collection = []
                help_collection = []
                for frame in range(frame_start,frame_end):
                    next_collection.append(frames_label[path][frame]['next'])
                    help_collection.append(frames_label[path][frame]['help'])
                next_label = max(set(next_collection), key=next_collection.count)
                help_label = max(set(help_collection), key=help_collection.count)
                activity_dataset[fl][index]['next_label'] = next_label
                activity_dataset[fl][index]['help'] = help_label
            
        return activity_dataset, frames_label, label_to_id, id_to_label

    def compute_frame_label(self, activity_dataset, help_dataset, dataset_path, label_to_id):
            collection = {}
            iter_count = 0
            files_path = {}
            for root, dirs, files in os.walk(dataset_path):
                for fl in files:
                    path = root + '/' + fl
                    if path in activity_dataset.keys():
                        files_path[path] = path
                    elif fl in activity_dataset.keys():
                        files_path[fl] = path

            # pp.pprint(files_path)
            pbar = tqdm(total=(len(files_path)), leave=False, desc='Generating Frame')

            for entry in files_path:
                path = files_path[entry]
                video = cv2.VideoCapture(path)
                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 2)
                fps = video.get(cv2.CAP_PROP_FPS)
                tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_label = {}
                for frame in range(1, tot_frames):
                    cut_name = path.split('/')[-1]
                    cut_name = cut_name.split('cam')[0]
                    obj = {}
                    if cut_name in self.object_label:
                        if frame in self.object_label[cut_name]:
                            obj_list = self.object_label[cut_name][frame]['object_list']
                            prob_list = self.object_label[cut_name][frame]['scores']
                            for indx in range(len(obj_list)):
                                obj[self.obj_id_2_label[obj_list[indx]]] = prob_list[indx]
                    else:
                        obj = {}
                    
                    frame_in_msec = (frame / float(fps)) * 1000
                    label = 'sil'
                    labels = {'now': label_to_id[label], 'next': label_to_id[label], 'help': label_to_id[label], 'obj' : obj}
                    for annotation in activity_dataset[entry]:
                        segment = annotation['milliseconds']
                        if frame_in_msec <= segment[1] and frame_in_msec >= segment[0]:
                            if annotation['label'] in list(label_to_id.keys()):
                                labels['now'] = label_to_id[annotation['label']]
                            break
                    for annotation in help_dataset[entry]:
                        segment = annotation['milliseconds']
                        if frame_in_msec <= segment[1] and frame_in_msec >= segment[0]:
                            if annotation['label'] in list(label_to_id.keys()):
                                labels['help'] = label_to_id[annotation['label']]
                            break
                    frames_label[frame] = labels
                for frame in range(1, tot_frames):
                    current_label = frames_label[frame]['now']
                    find_next = True
                    next_frame = frame + 1
                    next_action = label_to_id['sil']
                    next_maybe = label_to_id['sil']
                    while find_next and next_frame < tot_frames:
                        proposed_next_label = frames_label[next_frame]['now']
                        next_frame += + 1
                        # if proposed_next_label != current_label and proposed_next_label != label_to_id['sil']:
                        #     if proposed_next_label != label_to_id['giveobj'] and proposed_next_label != label_to_id['requestobj']:
                        #         next_action = proposed_next_label
                        #         find_next = False
                        #     else:
                        #         next_maybe = proposed_next_label
                        #     if next_maybe != proposed_next_label:
                        #         next_action = proposed_next_label
                        #         find_next = False
                        if proposed_next_label != current_label and proposed_next_label != label_to_id['sil']:
                            next_action = proposed_next_label
                            find_next = False
                    frames_label[frame]['next'] = next_action
                path = path.replace('\\','/')
                collection[path] = {}
                collection[path] = frames_label
                pbar.update(1)
            pbar.close()
            return collection

    def create_labels_mappings(self, label_collection):
        # label_collection.sort()
        label_to_id = {}
        id_to_label = {}
        label_to_id['sil'] = 0
        id_to_label[0] = 'sil'
        label_to_id['go'] = 1
        id_to_label[1] = 'go'
        label_to_id['end'] = 2
        id_to_label[2] = 'end'
        i = 3
        for label in label_collection:
            if label not in label_to_id:
                label_to_id[label] = i
                id_to_label[i] = label
                i += 1
        return label_to_id, id_to_label

    def clean_label(self, label, id):
        # print(label)
        if ':' in label:
            label = label.split(':')[1]
        # print(label)
        label = label.lower()
        
        # label = label.split(' ')
        return label