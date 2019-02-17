import tensorflow as tf
import numpy as np
import config
import pickle
import os


class tf_records_reader:
    def __init__(self):
        self.shape_X = [6, 112, 112, 7]
        with open('dataset/id_to_label.pkl', 'rb') as f:
            self.id_to_label = pickle.load(f)
        self.num_classes = len(self.id_to_label)

        self.filenames = ['./tf_records/milk/stir_milk-pour_milk/milk-stir_milk-pour_milk.tfrecord',
                          './tf_records/milk/take_cup-pour_milk/milk-take_cup-pour_milk.tfrecord',
                          './tf_records/milk/stir_milk-SIL/milk-stir_milk-SIL-0.tfrecord']


    def batch_generator(self):

        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self._parse_image_function)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size=config.Batch_size)

        iterator = dataset.make_one_shot_iterator()
        data = iterator.get_next()

        k = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                # Keep running next_batch till the Dataset is exhausted
                while True:
                    k = (sess.run(data))
            except tf.errors.OutOfRangeError:
                pass

        return k

    def _parse_image_function(self, example_proto):

        # Create a dictionary describing the features.
        image_feature_description = {
            'X': tf.FixedLenFeature(shape=[np.prod(self.shape_X)], dtype=tf.float32),
            'Y': tf.FixedLenFeature(shape=[self.num_classes], dtype=tf.int64),
            'next_Y': tf.FixedLenFeature(shape=[self.num_classes], dtype=tf.int64),
            'multi_next_Y': tf.FixedLenFeature(shape=[self.num_classes], dtype=tf.int64),
            'segment_collection': tf.FixedLenFeature(shape=[2], dtype=tf.int64),
            'video_name_collection': tf.FixedLenFeature(shape=[1], dtype=tf.string)
        }
        # Parse the input tf.Example proto using the dictionary above.
        sample = tf.parse_single_example(example_proto, image_feature_description)

        image = sample['X']
        image.set_shape(np.prod(self.shape_X))
        sample['X'] = tf.reshape(image, self.shape_X)

        X = sample['X']
        Y = sample['Y']
        next_Y = sample['next_Y']
        multi_next_Y = sample['multi_next_Y']
        segment_collection = sample['segment_collection']
        video_name_collection = sample['video_name_collection']

        return X, Y, next_Y, multi_next_Y, segment_collection, video_name_collection


func = tf_records_reader()

# the last batch
k = func.batch_generator()
# shape of X values
print(k[0].shape)
# video_name_collection values
print(k[5])
