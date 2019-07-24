import json
import config
import os
import pprint
from tqdm import tqdm
import cv2
import pickle
import random
pp = pprint.PrettyPrinter(indent=4)

def load(name):
        with open('dataset/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

ordered_collection = load('ordered_collection')

human_take = 0
human_video = 0
robot_take = 0
robot_video = 0
human_assistant_separation = {}
robot_assistant_separation = {}
for entry in ordered_collection.keys():
    is_robot = False
    path = entry
    if 'robot' in path:
        is_robot = True
    path = path.split('/')
    video_name = path[-1]
    video_name = video_name.split('_cam')
    take = video_name[0]
    cam = video_name[1]
    cam = cam.split('.')[0]
    if is_robot:
        if take not in robot_assistant_separation:
            robot_assistant_separation[take] = []
            robot_take += 1
        robot_assistant_separation[take].append(cam)
        robot_video += 1
    else:
        if take not in human_assistant_separation:
            human_assistant_separation[take] = []
            human_take += 1
        human_assistant_separation[take].append(cam)
        human_video += 1

pp.pprint(robot_assistant_separation)

print('human take:', human_take)
print('human video:', human_video)
print('human available video:', human_video - human_take)

print('robot take:', robot_take)
print('robot video:', robot_video)
print('robot available video:', robot_video - robot_take)
