import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import pickle
from tf_records_generator import IO_manager
from dataset_manager import Dataset
import config


class tf_writer:
    def __init__(self):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_CPP_MIN_LOG_LEVEL
        self.tf_config = tf.ConfigProto(inter_op_parallelism_threads=config.inter_op_parallelism_threads)
        self.tf_config.gpu_options.allow_growth = config.allow_growth

        Dataset()
        if config.limit_classes:
            self.activity_name = '+'.join(config.classes_to_use)
        else:
            self.activity_name = "whole_dataset"

        with open('dataset/train_collection.pkl', 'rb') as f:
            self.train_collection = pickle.load(f)
        with open('dataset/id_to_label.pkl', 'rb') as f:
            self.id_to_label = pickle.load(f)

        self.num_classes = len(self.id_to_label)
        list_couple = list(self.train_collection)
        self.Openpose_ran = False
        self.main(list_couple)




    def main(self, list_couple):

        for couple in list_couple:

            folder_name = self.id_to_label[int(couple.split('-')[0])] + '-' + self.id_to_label[
                int(couple.split('-')[1])]
            if not os.path.exists('./tf_records/' + self.activity_name + '/' + folder_name + '/'):
                os.makedirs('./tf_records/' + self.activity_name + '/' + folder_name + '/')


            if len(self.train_collection[couple]) < 20:
                print(couple)
                couple_name = folder_name
                input = [self.train_collection[couple]]
                final_data, length_data = self.extracting_data(input)
                self.tf_record_writer(final_data, length_data, couple_name,folder_name)

            elif len(self.train_collection[couple]) < 400:
                print(couple)
                couple_name = folder_name
                # spliting list chunks with size 20 for multiprocessing
                input = list(self.chunks_func(self.train_collection[couple]))
                final_data, length_data = self.extracting_data(input)
                self.tf_record_writer(final_data, length_data, couple_name, folder_name)

            else:
                i = 0
                total_input = list(self.chunks_func(self.train_collection[couple], n = 400))
                for chunk in total_input:
                    print(i, "out of ", len(total_input))
                    couple_name = folder_name + '-' + str(i)
                    input = list(self.chunks_func(chunk))
                    final_data, length_data = self.extracting_data(input)
                    self.tf_record_writer(final_data, length_data, couple_name, folder_name)
                    self.Openpose_ran = True
                    i +=1

            self.Openpose_ran = True


    def chunks_func(self, l, n = 20):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i+n]

    def extracting_data(self, couple_values):

        with tf.Session(config=self.tf_config) as sess:
            IO_tool = IO_manager(sess)
            total = 0
            for i in couple_values:
                total += len(i)
            if not self.Openpose_ran:
                IO_tool.start_openPose()
                IO_tool.openpose.load_openpose_weights()
            else:
                pass

            pbar = tqdm(total=(total * config.frames_per_step + len(couple_values)), leave=False,
                        desc='Batch Generation')
            ready_dataset = IO_tool.compute_batch(pbar, couple_values, total, self.num_classes, Train=True,
                                                  augment=True)
            large_data = self.recollecting_data(ready_dataset)

            return large_data, total

    def recollecting_data(self, ready_dataset):

        video_name_collection = []
        segment_collection = []
        multiple_next_label = []
        Y = []
        X = []
        next_Y = []
        for small_data in ready_dataset:
            video_name_collection.extend(small_data.get("video_name_collection"))
            segment_collection.extend(small_data.get("segment_collection"))
            multiple_next_label.extend(small_data.get("multi_next_Y"))
            Y.extend(small_data.get('Y'))
            X.extend(small_data.get('X'))
            next_Y.extend(small_data.get('next_Y'))

        large_data = {
            'X': np.array(X),
            'Y': np.array(Y),
            "next_Y": np.array(next_Y),
            'multi_next_Y': np.array(multiple_next_label),
            "video_name_collection": video_name_collection,
            "segment_collection": np.array(segment_collection)}

        #print(len(video_name_collection))
        return large_data

    def tf_record_writer(self, final_data, length_data, couple_name, folder_name):
        def get_example_object(i):
            # Convert individual data into a list of int64 or float or bytes
            def _bytes_feature(value):
                """Returns a bytes_list from a string / byte."""
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

            def _float_feature(value):
                """Returns a float_list from a float / double."""
                return tf.train.Feature(float_list=tf.train.FloatList(value=value))

            def _int64_feature(value):
                """Returns an int64_list from a bool / enum / int / uint."""
                return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

            feature_key_value_pair = {
                'X': _float_feature(final_data['X'][i].reshape(-1)),
                'Y': _int64_feature(final_data['Y'][i].reshape(-1)),
                'next_Y': _int64_feature(final_data['next_Y'][i].reshape(-1)),
                'multi_next_Y': _int64_feature(final_data['multi_next_Y'][i].reshape(-1)),
                'segment_collection': _int64_feature(final_data['segment_collection'][i].reshape(-1)),
                'video_name_collection': _bytes_feature([final_data['video_name_collection'][i].encode('utf-8')])

            }

            # Create Features object with above feature dictionary
            features = tf.train.Features(feature=feature_key_value_pair)
            # Create Example object with features
            example = tf.train.Example(features=features)

            # print(feature_key_value_pair['X'].shape)
            return example

        with tf.python_io.TFRecordWriter('./tf_records/' + self.activity_name+'/' + folder_name + '/'
                                         + self.activity_name + "-" +couple_name + '.tfrecord') as tfwriter:
            # Iterate through all records
            for i in range(length_data):
                example = get_example_object(i)
                # Append each example into tfrecord
                tfwriter.write(example.SerializeToString())


tf_writer()
