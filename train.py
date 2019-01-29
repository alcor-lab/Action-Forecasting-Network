import tensorflow as tf
import os
import multiprocessing.dummy as mp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pprint
from batch_generator import IO_manager
from network import activity_network
from network import Training
from network import Input_manager
import config
from tensorflow.python.client import device_lib

pp = pprint.PrettyPrinter(indent=4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_CPP_MIN_LOG_LEVEL
tf_config = tf.ConfigProto(inter_op_parallelism_threads=config.inter_op_parallelism_threads, allow_soft_placement = True)
tf_config.gpu_options.allow_growth = config.allow_growth

def calculate_confusion_matrix(confusion, batch, y_pred, step, folder_name, id_to_label):
    if not os.path.exists('./results/confusion_matrix/' + folder_name + '/'):
        os.makedirs('./results/confusion_matrix/' + folder_name + '/')

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
            plt.savefig('./results/confusion_matrix/' + folder_name + '/cm' + str(step) + '.png')
        else:
            plt.savefig('./results/confusion_matrix/' + folder_name + '/cm.png')
        plt.gcf().clear()
    return confusion

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos if x.device_type == 'GPU']
    # return local_device_protos

def train():
    with tf.Session(config=tf_config) as sess:
        IO_tool = IO_manager(sess)
        number_of_classes = IO_tool.num_classes
        available_gpus = get_available_gpus()
        j=0
        Net_collection = {}
        Input_net = Input_manager(len(available_gpus))
        for device in available_gpus:
            with tf.device(device.name):
                print(device.name)
                with tf.variable_scope('Network') as scope:
                    if j>0:
                        scope.reuse_variables()
                    Net_collection['Network_' + str(j)] = activity_network(number_of_classes, Input_net, j)
                    j = j+1
        Train_Net = Training(Net_collection)
        IO_tool.start_openPose()
        train_writer = tf.summary.FileWriter("logdir/train", sess.graph)
        val_writer = tf.summary.FileWriter("logdir/val", sess.graph)
        IO_tool.openpose.load_openpose_weights()
        sess.run(Train_Net.init)



        # Loading initial C3d or presaved network
        config.load_pretrained_weigth = True
        if os.path.isfile('./checkpoint/checkpoint') and config.load_pretrained_weigth:
            print('new model loaded')
            Net_collection['Network_0'].model_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
        elif config.load_previous_weigth and config.load_pretrained_weigth:
            print('original c3d loaded')
            Net_collection['Network_0'].c3d_loader.restore(sess, config.c3d_ucf_weights)

        confusion_train_Lstm = np.zeros((number_of_classes, number_of_classes))
        confusion_train_C3d = np.zeros((number_of_classes, number_of_classes))
        confusion_train_Next = np.zeros((number_of_classes, number_of_classes))
        confusion_val_Lstm = np.zeros((number_of_classes, number_of_classes))
        confusion_val_C3d = np.zeros((number_of_classes, number_of_classes))
        confusion_val_Next = np.zeros((number_of_classes, number_of_classes))
        step = 0
        training = True
        with tf.name_scope('whole_saver'):
            whole_saver = tf.train.Saver()
        minimum_trimmed = config.snow_ball_per_class * number_of_classes
        pbar_whole = tqdm(total=(config.tot_steps), desc='Step')
        while step < config.tot_steps:
            ready_batch = 0
            pbar = tqdm(total=(config.tasks * config.Batch_size * len(available_gpus) * config.frames_per_step + len(available_gpus)*config.tasks - 1), leave=False, desc='Batch Generation')
            ready_batch = IO_tool.compute_batch(pbar, Devices=len(available_gpus), Train=training)
            for batch in ready_batch:
                summary, t_op, y_Lstm, y_c3d, c_state, h_state, y_Next = sess.run([Train_Net.merged, Train_Net.train_op,
                                                                           Train_Net.predictions_Lstm_list, Train_Net.predictions_c3d_list,
                                                                           Train_Net.c_out_list, Train_Net.h_out_list,
                                                                           Train_Net.predictions_Lstm_next_list],
                                                                          feed_dict={Input_net.input_batch: batch['X'],
                                                                                     Input_net.labels: batch['Y'],
                                                                                     Input_net.c_input: batch['c'],
                                                                                     Input_net.h_input: batch['h'],
                                                                                     Input_net.next_labels: batch['next_Y'],
                                                                                     Input_net.multiple_next_labels: batch['multi_next_Y']})

                for j in range(len(batch['video_name_collection'])):
                    for y in range(c_state[0].shape[0]):
                        IO_tool.add_hidden_state(batch['video_name_collection'][j][y],
                                                batch['segment_collection'][j][y][1],
                                                h_state[j][y],
                                                c_state[j][y])
                    confusion_train_C3d = calculate_confusion_matrix(confusion_train_C3d, batch['Y'][j], y_c3d[j], (step + 1) * config.Batch_size, 'c3d', IO_tool.dataset.id_to_label)
                    confusion_train_Lstm = calculate_confusion_matrix(confusion_train_Lstm, batch['Y'][j], y_Lstm[j], (step + 1) * config.Batch_size, 'lstm', IO_tool.dataset.id_to_label)
                    confusion_train_Next = calculate_confusion_matrix(confusion_train_Next, batch['Y'][j], y_Next[j], (step + 1) * config.Batch_size, 'next', IO_tool.dataset.id_to_label)

                step = step + config.Batch_size*len(available_gpus)
                train_writer.add_summary(summary, step)
                pbar_whole.update(config.Batch_size*len(available_gpus))

                if step % 1000 == 0 or (step + 1) == config.tot_steps:
                    validation = True
                    if validation:
                        val_step = step
                        pbar_val = tqdm(total=(config.tasks * config.Batch_size * len(available_gpus) * config.frames_per_step + len(available_gpus)*config.tasks - 1), leave=False, desc='Validation Generation')

                        val_batch = IO_tool.compute_batch(pbar_val, Devices=len(available_gpus), Train=False, augment=False)

                        for batch in val_batch:
                            summary, y_Lstm, y_c3d, c_state, h_state, y_Next = sess.run([Train_Net.merged,
                                                                                 Train_Net.predictions_Lstm_list, Train_Net.predictions_c3d_list,
                                                                                 Train_Net.c_out_list, Train_Net.h_out_list,
                                                                                 Train_Net.predictions_Lstm_next_list],
                                                                                feed_dict={Input_net.input_batch: batch['X'],
                                                                                           Input_net.labels: batch['Y'],
                                                                                           Input_net.c_input: batch['c'],
                                                                                           Input_net.h_input: batch['h'],
                                                                                           Input_net.next_labels: batch['next_Y'],
                                                                                           Input_net.multiple_next_labels: batch['multi_next_Y']})
                            for j in range(len(batch['video_name_collection'])):
                                for y in range(c_state[0].shape[0]):
                                    IO_tool.add_hidden_state(batch['video_name_collection'][j][y],
                                                            batch['segment_collection'][j][y][1],
                                                            h_state[j][y],
                                                            c_state[j][y])

                                confusion_val_C3d = calculate_confusion_matrix(confusion_val_C3d, batch['Y'][j], y_c3d[j], (step + 1) * config.Batch_size, 'c3d_val', IO_tool.dataset.id_to_label)
                                confusion_val_Lstm = calculate_confusion_matrix(confusion_val_Lstm, batch['Y'][j], y_Lstm[j], (step + 1) * config.Batch_size, 'lstm_val', IO_tool.dataset.id_to_label)
                                confusion_val_Next = calculate_confusion_matrix(confusion_val_Next, batch['Y'][j], y_Next[j], (step + 1) * config.Batch_size, 'next_val', IO_tool.dataset.id_to_label)
                            val_writer.add_summary(summary, val_step + config.Batch_size*len(available_gpus))
                            val_step += 1

                    IO_tool.save_hidden_state_collection()
                    # IO_tool.hidden_states_statistics()
                    confusion_train_Lstm = np.zeros((number_of_classes, number_of_classes))
                    confusion_train_C3d = np.zeros((number_of_classes, number_of_classes))
                    confusion_train_Next = np.zeros((number_of_classes, number_of_classes))
                    confusion_val_Lstm = np.zeros((number_of_classes, number_of_classes))
                    confusion_val_C3d = np.zeros((number_of_classes, number_of_classes))
                    confusion_val_Next = np.zeros((number_of_classes, number_of_classes))
                    whole_saver.save(sess, config.model_filename, global_step=step)


train()
