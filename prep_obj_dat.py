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
for root, dirs, files in os.walk('dataset/object_label'):
    for fl in files:
        if fl.split('.')[-1] == 'json' and 'trial' in fl.split('.')[0]:
            path = root +  '/' + fl
            print(path)
            json_data = open(path).read()
            Dataset = json.loads(json_data)
            obj_dataset.update(Dataset)
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
for folder in obj_dataset:
    if folder in folder_to_name:
        name = folder_to_name[folder]
        cut_video_name = name.split('cam')[0]
        new_collection_video_name[cut_video_name] = obj_dataset[folder]
    else:
        new_collection_video_name[folder] = obj_dataset[folder]

pp.pprint(list(new_collection_video_name.keys()))

# ordered_collection = load('ordered_collection')

# new_collection = {}
# for path in ordered_collection:
#     full_video_name = path.split('/')[-1]
#     cut_video_name = full_video_name.split('cam')[0]
#     if cut_video_name not in new_collection_video_name:
#         print(cut_video_name)
#     else:
#         new_collection[cut_video_name] = new_collection_video_name[cut_video_name]

with open(config.kit_obj_annotation, 'w') as outfile:
    json.dump(new_collection_video_name, outfile)

# pp.pprint(help_dataset)