import json
import config
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

help_dataset = {}
for root, dirs, files in os.walk('dataset'):
    for fl in files:
        if fl.split('.')[1] == 'json' and 'kit_activity' in fl.split('.')[0]:
            path = root +  '/' + fl
            print(path)
            json_data = open(path).read()
            Dataset = json.loads(json_data)
            help_dataset.update(Dataset)
            print(len(help_dataset))

with open(config.kit_activity_annotation_join, 'w') as outfile:
    json.dump(help_dataset, outfile)

# pp.pprint(help_dataset)