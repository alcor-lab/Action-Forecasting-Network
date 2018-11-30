import tensorflow as tf
import os
import multiprocessing.dummy as mp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pprint
from batch_generator import IO_manager
from network import activity_network


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

c3d_saved_weigths = "./checkpoint/sports1m_finetuning_ucf101.model"
number_of_classes = 16
C3d_number_of_frames = 16
C3d_height = 112
C3d_weigth = C3d_height
pp = pprint.PrettyPrinter(indent=4)

mode = tf.estimator.ModeKeys.TRAIN

config = tf.ConfigProto(inter_op_parallelism_threads=7)
config.gpu_options.allow_growth = True

learning_rate = 0.1
gradient_clipping_norm = 1.0
c3d_dropout = 0.6
preLstm_dropout = 0.6
Lstm_dropout = 0.6
C3d_CHANNELS = 7
C3d_Output_features = 250
lstm_units = C3d_Output_features * 2

Batch_size = 20
frames_per_step = 6
window_size = 1
height = 112
width = height * 1

load_c3d = True
use_pretrained_model = True
model_filename = './checkpoint/Net_weigths.model'
Tot_steps = 1000000

tasks = 10


def calculate_confusion_matrix(confusion, batch, y_pred, step, folder_name, id_to_label):
    if not os.path.exists('./confusion_matrix/' + folder_name + '/'):
        os.makedirs('./confusion_matrix/' + folder_name + '/')

    y_true = np.argmax(batch, axis=1)
    shape_label = y_true.shape
    for i in range(shape_label[0]):
        true_label = y_true[i]
        actual_label = y_pred[i]
        confusion[true_label, actual_label] += 1

    if (step) % 100 == 0 or folder_name in ['c3d_val', 'lstm_val']:
        if (step) % 1000 == 0:
            nm_confusion = confusion / confusion.astype(np.float).sum(axis=1)
            plt.imshow(nm_confusion, cmap=plt.cm.Blues, aspect='auto')
        else:
            plt.imshow(confusion, cmap=plt.cm.Blues, aspect='auto')
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        dict_len = len(id_to_label.keys())
        label_list = [None] * dict_len
        for i in id_to_label:
            try:
                label_list[i] = id_to_label[i].split(':')[1]
            except:
                label_list[i] = id_to_label[i]
        plt.yticks(np.arange(dict_len), label_list)
        plt.yticks([], [])
        plt.xticks([], [])
        plt.title('Confusion matrix Step:' + str(step))
        plt.colorbar()
        if (step) % 1000 == 0:
            plt.savefig('./confusion_matrix/' + folder_name + '/cm' + str(step) + '.png')
        else:
            plt.savefig('./confusion_matrix/' + folder_name + '/cm.png')
        plt.gcf().clear()
    return confusion


def train():
    with tf.Session(config=config) as sess:
        Network = activity_network(number_of_classes=49)
        IO_tool = IO_manager(Batch_size, frames_per_step, window_size, sess)
        number_of_classes = IO_tool.num_classes
        IO_tool.openpose.load_openpose_weights()
        IO_tool.hidden_states_dim = Network.lstm_units
        sess.run(Network.init)
        trimmed = True

        train_writer = tf.summary.FileWriter("logdir/train", sess.graph)
        val_writer = tf.summary.FileWriter("logdir/val", sess.graph)

        # Loading initial C3d or presaved network
        use_pretrained_model = True
        if os.path.isfile('./checkpoint/checkpoint') and use_pretrained_model:
            print('new model loaded')
            Network.model_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
        elif load_c3d and use_pretrained_model:
            print('original c3d loaded')
            Network.c3d_loader.restore(sess, c3d_saved_weigths)

        step = 0
        confusion_train_Lstm = np.zeros((number_of_classes, number_of_classes))
        confusion_train_C3d = np.zeros((number_of_classes, number_of_classes))
        confusion_train_Next = np.zeros((number_of_classes, number_of_classes))
        confusion_val_Lstm = np.zeros((number_of_classes, number_of_classes))
        confusion_val_C3d = np.zeros((number_of_classes, number_of_classes))
        confusion_val_Next = np.zeros((number_of_classes, number_of_classes))
        training = True
        Action = True
        with tf.name_scope('whole_saver'):
            whole_saver = tf.train.Saver()
        whole_saver.save(sess, model_filename, global_step=Network.global_step)
        minimum_trimmed = IO_tool.snow_ball_per_class * number_of_classes
        pbar_whole = tqdm(total=(Tot_steps), desc='Step')
        while step < minimum_trimmed * 5:
            ready_batch = 0
            if training:
                if step * Batch_size >= minimum_trimmed:
                    trimmed = False
                pbar = tqdm(total=(tasks * Batch_size * frames_per_step + tasks), leave=False, desc='Batch Generation')

                def multiprocess_batch(x):
                    X, Y, c, h, video_name_collection, segment_collection, next_label = IO_tool.batch_generator(pbar, Train=training, Trimmed=trimmed, Action=Action)
                    return {'X': X, 'Y': Y, 'c': c, 'h': h,
                            'video_name_collection': video_name_collection,
                            'segment_collection': segment_collection,
                            'next_Y': next_label}

                pool = mp.Pool(processes=tasks)
                ready_batch = pool.map(multiprocess_batch, range(0, tasks))
                pbar.close()
                pool.close()
                pool.join()
                ready_batch = IO_tool.add_pose(ready_batch, sess)

                for batch in ready_batch:
                    summary, t_op, y_Lstm, y_c3d, c_state, h_state, y_Next = sess.run([Network.merged, Network.train_op,
                                                                               Network.predictions_Lstm, Network.predictions_c3d,
                                                                               Network.c_out, Network.h_out,
                                                                               Network.predictions_Lstm_next],
                                                                              feed_dict={Network.input_batch: batch['X'],
                                                                                         Network.labels: batch['Y'],
                                                                                         Network.c_input: batch['c'],
                                                                                         Network.h_input: batch['h'],
                                                                                         Network.next_labels: batch['next_Y']})

                    for y in range(c_state.shape[0]):
                        IO_tool.add_hidden_state(batch['video_name_collection'][y],
                                                 batch['segment_collection'][y][1],
                                                 h_state[y],
                                                 c_state[y])

                    confusion_train_C3d = calculate_confusion_matrix(confusion_train_C3d, batch['Y'], y_c3d, (step + 1) * Batch_size, 'c3d', IO_tool.dataset.id_to_label)
                    confusion_train_Lstm = calculate_confusion_matrix(confusion_train_Lstm, batch['Y'], y_Lstm, (step + 1) * Batch_size, 'lstm', IO_tool.dataset.id_to_label)
                    confusion_train_Next = calculate_confusion_matrix(confusion_train_Next, batch['Y'], y_Next, (step + 1) * Batch_size, 'next', IO_tool.dataset.id_to_label)

                    step = step + 1
                    Action = not Action
                    train_writer.add_summary(summary, (step) * Batch_size)
                    pbar_whole.update(Batch_size)

                    real_step = (step) * Batch_size
                    if real_step % 10000 == 0 and real_step > 0:
                        IO_tool.dataset.generate_dataset()
                    if real_step % 1000 == 0 or (real_step + 1) == Tot_steps:
                        validation = True
                        if validation:
                            val_step = step
                            pbar_val = tqdm(total=(tasks * Batch_size * frames_per_step + tasks), leave=False, desc='Validation Generation')

                            def validation_batch(x):
                                X, Y, c, h, video_name_collection, segment_collection, next_label = IO_tool.batch_generator(pbar_val, Train=False, Trimmed=trimmed, Action=Action)
                                return {'X': X, 'Y': Y, 'c': c, 'h': h,
                                        'video_name_collection': video_name_collection,
                                        'segment_collection': segment_collection,
                                        'next_Y': next_label}

                            pool = mp.Pool(processes=tasks)
                            ready_batch = pool.map(validation_batch, range(0, tasks))
                            pbar_val.close()
                            pool.close()
                            pool.join()
                            val_batch = IO_tool.add_pose(ready_batch, sess, augment=False)

                            for batch in val_batch:
                                summary, y_Lstm, y_c3d, c_state, h_state, y_Next = sess.run([Network.merged,
                                                                                     Network.predictions_Lstm, Network.predictions_c3d,
                                                                                     Network.c_out, Network.h_out,
                                                                                     Network.predictions_Lstm_next],
                                                                                    feed_dict={Network.input_batch: batch['X'],
                                                                                               Network.labels: batch['Y'],
                                                                                               Network.c_input: batch['c'],
                                                                                               Network.h_input: batch['h'],
                                                                                               Network.next_labels: batch['next_Y']})
                                for y in range(c_state.shape[0]):
                                    IO_tool.add_hidden_state(batch['video_name_collection'][y],
                                                             batch['segment_collection'][y][1],
                                                             h_state[y],
                                                             c_state[y])

                                confusion_val_C3d = calculate_confusion_matrix(confusion_val_C3d, batch['Y'], y_c3d, (step + 1) * Batch_size, 'c3d_val', IO_tool.dataset.id_to_label)
                                confusion_val_Lstm = calculate_confusion_matrix(confusion_val_Lstm, batch['Y'], y_Lstm, (step + 1) * Batch_size, 'lstm_val', IO_tool.dataset.id_to_label)
                                confusion_val_Next = calculate_confusion_matrix(confusion_val_Next, batch['Y'], y_Next, (step + 1) * Batch_size, 'next_val', IO_tool.dataset.id_to_label)
                                val_writer.add_summary(summary, (val_step + 1) * Batch_size)
                                val_step += 1

                        IO_tool.save_hidden_state_collection()
                        IO_tool.hidden_states_statistics()
                        confusion_train_Lstm = np.zeros((number_of_classes, number_of_classes))
                        confusion_train_C3d = np.zeros((number_of_classes, number_of_classes))
                        confusion_train_Next = np.zeros((number_of_classes, number_of_classes))
                        confusion_val_Lstm = np.zeros((number_of_classes, number_of_classes))
                        confusion_val_C3d = np.zeros((number_of_classes, number_of_classes))
                        confusion_val_Next = np.zeros((number_of_classes, number_of_classes))
                        whole_saver.save(sess, model_filename, global_step=Network.global_step)


train()
