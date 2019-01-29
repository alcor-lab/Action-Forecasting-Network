import json
import config
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

class Annotation:
    def __init__(self):
        if config.dataset is 'Ocado':
            json_data = open(config.ocado_annotation).read()
            Dataset = json.loads(json_data)
        elif config.dataset is 'Breakfast':
            self.create_breakfast_annotation()
            json_data = open(config.breakfast_annotation).read()
            Dataset = json.loads(json_data)
        elif config.dataset is 'ActivityNet':
            json_data = open(config.acitivityNet_annotation).read()
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

        if config.limit_classes:
            new_Dataset = {}
            for video in Dataset.keys():
                for annotation in Dataset[video]:
                    if annotation['activity'] not in config.classes_to_use:
                        continue
                    if video not in new_Dataset:
                        new_Dataset[video]=[]
                    new_Dataset[video].append(annotation)
            self.Dataset = new_Dataset
        else:
            self.Dataset = Dataset

    def create_breakfast_annotation(self):
        Final_Annotation = {}
        for root, dirs, files in os.walk(config.breakfast_path):
            for fl in files:
                path = root + '/' + fl
                if fl[-4:] == ".txt":
                    f = open(path)
                    text = f.readlines()
                    Activity = fl.split('.')[0].split('_')[1]
                    listaOut = []
                    for line in text:
                        temp_dict = {}
                        annotation = line.strip().split(' ')
                        if len(annotation) != 2:
                            break
                        temp_dict['label'] = annotation[1]
                        frames = annotation[0].strip().split('-')
                        milliseconds=[]
                        for e in frames:
                            milliseconds.append(int((int(e)/float(config.breakfast_fps))*1000))
                        temp_dict['milliseconds'] = milliseconds
                        temp_dict['activity'] = Activity
                        listaOut.append(temp_dict)
                    for index in range(len(listaOut) - 1 ):
                        listaOut[index]['next_label'] = listaOut[index + 1]['label']
                    listaOut[-1]['next_label'] = 'END'
                    entry_name = str(path).split('.')[0]+'.avi'
                    Final_Annotation[entry_name] = listaOut
        with open(config.breakfast_annotation, 'w') as outfile:
            json.dump(Final_Annotation, outfile)
