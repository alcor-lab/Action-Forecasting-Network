import json
import config
import os
import pprint
from tqdm import tqdm
import cv2
pp = pprint.PrettyPrinter(indent=4)

json_data = open(config.kit_help_annotation_temp).read()
help_dataset = json.loads(json_data)


map_1 = {
    'get cloth from technician on ladder and put on the table': 'get_from_technician_and_put_on_the_table cloth on_ladder',
    'get cloth from technician under diverter and put on the table': 'get_from_technician_and_put_on_the_table cloth under_diverter',
    'get spray-bottle from technician near guard-support and put on the table': 'get_from_technician_and_put_on_the_table spray_bottle guard_support',
    'get spray-bottle from technician on ladder and put on the table': 'get_from_technician_and_put_on_the_table spray_bottle on_ladder',
    'get spray-bottle from technician under diverter and put on the table': 'get_from_technician_and_put_on_the_table spray_bottle under_diverter',
    'get torch from technician near guard-support and put on the table': 'get_from_technician_and_put_on_the_table torch guard_support',
    'get torch from technician under diverter and put on the table': 'get_from_technician_and_put_on_the_table torch under_diverter',
    'get spray-bottle from technician near guard and put on the table':'get_from_technician_and_put_on_the_table spray_bottle guard_support',
    'get torch from technician on ladder and put on the table':'get_from_technician_and_put_on_the_table torch on_ladder',
    'grasp guard and put on diverter': 'grasp_and_put_on_diverter guard at_guard_support',
    'remove guard and put down': 'remove_and_put_down guard under_diverter'

}

map_2 = {
    'give cloth to the technician sil' : 'give_to_technician cloth sil',
    'give cloth to the technician under_diverter' : 'give_to_technician cloth under_diverter',
    'give spray-bottle to the technician on_ladder' : 'give_to_technician spray_bottle on_ladder',
    'give spray-bottle to the technician sil' : 'give_to_technician spray_bottle sil',
    'give spray-bottle to the technician under_diverter' : 'give_to_technician spray_bottle under_diverter',
    'give torch to the technician on_ladder' : 'give_to_technician torch on_ladder',
    'give torch to the technician sil' : 'give_to_technician torch sil',
    'give torch to the technician under_diverter' : 'give_to_technician torch under_diverter',

}

def clean_label(label):
    label = label.split(':')
    if label[0] in map_1:
        label = map_1[label[0]]
    else:
        label = label[0] + ':' + label[1]
    return label

def clean_label_2(label, dat, fl):
    label = label.split(':')
    if len(label) > 1:
        if 'under diverter' in label[1]:
            label = label[0] + ' under_diverter'
        elif 'on ladder' in label[1]:
            label = label[0] + ' on_ladder'
        elif 'at guard-support' in label[1]:
            label = label[0] + ' at_guard_support'
        else:
            pp.pprint(fl)
            # pp.pprint(dat)
            print(label)
            label = label[0] + ' sil'
    else:
        label = label[0]
    return label

def clean_label_3(label):
    if label in map_2:
        label = map_2[label]
    return label

labels = []
label_collection = []
# pp.pprint(help_dataset['recording_03-15-2019_16-14-27.267_cam6.avi'])

for fl in help_dataset:
    label_list = []
    segment_start_to_entry = {}
    # print('NEW')
    # pp.pprint(help_dataset[fl])

    # translation next help into previous sil help
    pp.pprint(help_dataset[fl])

    for idx in range(len(help_dataset[fl])-1):        
        entry = help_dataset[fl][idx]
        if entry['label'].split(':')[1] == 'sil':
            if help_dataset[fl][idx+1]['label'].split(':')[0] != entry['label'].split(':')[0]:
                entry['label']=help_dataset[fl][idx+1]['label']

    pp.pprint(help_dataset[fl])

    #map start time to entry
    for idx in range(len(help_dataset[fl])):        
        entry = help_dataset[fl][idx]
        segment = entry['milliseconds']
        segment_start_to_entry[segment[0]] = entry

    #order entry by start frame
    new_entry_order = []
    while len(segment_start_to_entry) > 0:
        min_key = min(segment_start_to_entry.keys())
        new_entry_order.append(segment_start_to_entry[min_key])
        del segment_start_to_entry[min_key]
    help_dataset[fl] = new_entry_order

    #cut overlapping entry by start frame
    new_entry_order_2 = []
    for idx in range(len(new_entry_order) - 1):
        entry = new_entry_order[idx]
        segment = entry['milliseconds']
        next_segment = help_dataset[fl][idx+1]['milliseconds'][0]
        if segment[1] > next_segment:
            segment[1] = next_segment- 1
        new_entry_order[idx]['milliseconds'] = segment
        new_entry_order_2.append(new_entry_order[idx])

    # first label clean
    for idx in range(len(help_dataset[fl])):
        entry = help_dataset[fl][idx]
        label = entry['label']
        if label.split(':')[0] not in label_collection:
            label_collection.append(label.split(':')[0])
        label = clean_label(label)
        help_dataset[fl][idx]['label'] = label

    # join cleaned label
    new_entry_order = []
    idx = 0
    while idx < len(help_dataset[fl]) - 1:
        entry = help_dataset[fl][idx]
        while help_dataset[fl][idx +1]['label'] == entry['label']:
            entry['milliseconds'][1] =  help_dataset[fl][idx +1]['milliseconds'][1]
            del help_dataset[fl][idx +1]
            if (idx+1) > len(help_dataset[fl]) - 1:
                break
        new_entry_order.append(entry)
        idx += 1

    if help_dataset[fl][-1]['label'] != new_entry_order[-1]['label']:
        new_entry_order.append(help_dataset[fl][-1])

    help_dataset[fl] = new_entry_order

    # join uncleaned label
    new_entry_order = []
    idx = 0
    while idx < len(help_dataset[fl]) - 1:
        entry = help_dataset[fl][idx]
        while help_dataset[fl][idx +1]['label'].split(':')[0] == entry['label'].split(':')[0]:
            entry['label'] += (' ') + help_dataset[fl][idx +1]['label'].split(':')[1]
            entry['milliseconds'][1] =  help_dataset[fl][idx +1]['milliseconds'][1]
            del help_dataset[fl][idx +1]
            if (idx+1) > len(help_dataset[fl]) - 1:
                break
        new_entry_order.append(entry)
        idx += 1

    if help_dataset[fl][-1]['label'] != new_entry_order[-1]['label']:
        new_entry_order.append(help_dataset[fl][-1])

    help_dataset[fl] = new_entry_order

    # search location for uncleaned label
    # and create list of all labels
    for idx in range(len(help_dataset[fl])):
        entry = help_dataset[fl][idx]
        label = entry['label']
        label = clean_label_2(label, help_dataset[fl], fl)
        label = clean_label_3(label)
        help_dataset[fl][idx]['label'] = label
        if label not in labels:
            labels.append(label)
    
with open(config.kit_help_annotation, 'w') as outfile:
    json.dump(help_dataset, outfile)

label_collection.sort()
pp.pprint(label_collection)
labels.sort()
print('labels')
pp.pprint(labels)




