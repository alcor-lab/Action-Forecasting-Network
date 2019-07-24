import json
import config
import os
import pickle
import pprint
pp = pprint.PrettyPrinter(indent=4)

def load(name):
    with open('dataset/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


obj_dataset = {}
json_data = open('dataset/object_label/folder_to_video.json').read()
folder_to_name = json.loads(json_data)
for root, dirs, files in os.walk('dataset/location_label'):
    for fl in files:
        temp_dat = {}
        if fl.split('.')[-1] == 'json' and 'Trial' in fl.split('.')[0]:
            path = root +  '/' + fl
            print(path)
            json_data = open(path).read()
            Dataset = json.loads(json_data)
            temp_dat[fl] = Dataset
            obj_dataset.update(temp_dat)
            print(len(obj_dataset))
        if fl.split('.')[-1] == 'json' and 'record' in fl.split('.')[0]:
            path = root +  '/' + fl
            print(path)
            json_data = open(path).read()
            Dataset = json.loads(json_data)
            new_dat = {}
            val = Dataset[list(Dataset.keys())[0]]
            new_dat[fl.split('cam')[0]] = val
            obj_dataset.update(new_dat)
            print(len(obj_dataset))

new_collection_video_name = {}
for key in obj_dataset:
    folder = ('Trial' + key.split('Trial')[1]).lower().split('.')[0]
    if folder in folder_to_name:
        name = folder_to_name[folder]
        cut_video_name = name.split('cam')[0]
        new_collection_video_name[cut_video_name] = obj_dataset[key]
    else:
        new_collection_video_name[folder] = obj_dataset[key]

pp.pprint(list(new_collection_video_name.keys()))
print(folder_to_name)
with open(config.kit_loc_annotation, 'w') as outfile:
    json.dump(new_collection_video_name, outfile)
