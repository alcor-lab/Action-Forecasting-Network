import tensorflow as tf
import os
import multiprocessing.dummy as mp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pprint
from batch_generator import IO_manager
from network import activity_network
import config

pp = pprint.PrettyPrinter(indent=4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_CPP_MIN_LOG_LEVEL
tf_config = tf.ConfigProto(inter_op_parallelism_threads=config.inter_op_parallelism_threads)
tf_config.gpu_options.allow_growth = config.allow_growth

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
    with tf.Session(config=tf_config) as sess:
        IO_tool = IO_manager(sess)
        number_of_classes = IO_tool.num_classes
        Network = activity_network(number_of_classes)
        IO_tool.start_openPose()
        IO_tool.openpose.load_openpose_weights()
        sess.run(Network.init)
        trimmed = True

        train_writer = tf.summary.FileWriter("logdir/train", sess.graph)
        val_writer = tf.summary.FileWriter("logdir/val", sess.graph)

        # Loading initial C3d or presaved network
        config.load_pretrained_weigth = True
        if os.path.isfile('./checkpoint/checkpoint') and config.load_pretrained_weigth:
            print('new model loaded')
            Network.model_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
        elif config.load_previous_weigth and config.load_pretrained_weigth:
            print('original c3d loaded')
            Network.c3d_loader.restore(sess, config.c3d_ucf_weights)

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
        whole_saver.save(sess, config.model_filename, global_step=Network.global_step)
        minimum_trimmed = config.snow_ball_per_class * number_of_classes
        pbar_whole = tqdm(total=(config.tot_steps), desc='Step')
        while step < minimum_trimmed * 5:
            ready_batch = 0
            if training:
                if step * config.Batch_size >= minimum_trimmed:
                    trimmed = False
                pbar = tqdm(total=(config.tasks * config.Batch_size * config.frames_per_step + config.tasks), leave=False, desc='Batch Generation')

                def multiprocess_batch(x):
                    X, Y, c, h, video_name_collection, segment_collection, next_label = IO_tool.batch_generator(pbar, Train=training, Trimmed=trimmed, Action=Action)
                    return {'X': X, 'Y': Y, 'c': c, 'h': h,
                            'video_name_collection': video_name_collection,
                            'segment_collection': segment_collection,
                            'next_Y': next_label}

                pool = mp.Pool(processes=config.tasks)
                ready_batch = pool.map(multiprocess_batch, range(0, config.tasks))
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

                    confusion_train_C3d = calculate_confusion_matrix(confusion_train_C3d, batch['Y'], y_c3d, (step + 1) * config.Batch_size, 'c3d', IO_tool.dataset.id_to_label)
                    confusion_train_Lstm = calculate_confusion_matrix(confusion_train_Lstm, batch['Y'], y_Lstm, (step + 1) * config.Batch_size, 'lstm', IO_tool.dataset.id_to_label)
                    confusion_train_Next = calculate_confusion_matrix(confusion_train_Next, batch['Y'], y_Next, (step + 1) * config.Batch_size, 'next', IO_tool.dataset.id_to_label)

                    step = step + 1
                    Action = not Action
                    train_writer.add_summary(summary, (step) * config.Batch_size)
                    pbar_whole.update(config.Batch_size)

                    real_step = (step) * config.Batch_size
                    if real_step % 10000 == 0 and real_step > 0:
                        IO_tool.dataset.generate_dataset()
                    if real_step % 1000 == 0 or (real_step + 1) == config.tot_steps:
                        validation = True
                        if validation:
                            val_step = step
                            pbar_val = tqdm(total=(config.tasks * config.Batch_size * config.frames_per_step + config.tasks), leave=False, desc='Validation Generation')

                            def validation_batch(x):
                                X, Y, c, h, video_name_collection, segment_collection, next_label = IO_tool.batch_generator(pbar_val, Train=False, Trimmed=trimmed, Action=Action)
                                return {'X': X, 'Y': Y, 'c': c, 'h': h,
                                        'video_name_collection': video_name_collection,
                                        'segment_collection': segment_collection,
                                        'next_Y': next_label}

                            pool = mp.Pool(processes=config.tasks)
                            ready_batch = pool.map(validation_batch, range(0, config.tasks))
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

                                confusion_val_C3d = calculate_confusion_matrix(confusion_val_C3d, batch['Y'], y_c3d, (step + 1) * config.Batch_size, 'c3d_val', IO_tool.dataset.id_to_label)
                                confusion_val_Lstm = calculate_confusion_matrix(confusion_val_Lstm, batch['Y'], y_Lstm, (step + 1) * config.Batch_size, 'lstm_val', IO_tool.dataset.id_to_label)
                                confusion_val_Next = calculate_confusion_matrix(confusion_val_Next, batch['Y'], y_Next, (step + 1) * config.Batch_size, 'next_val', IO_tool.dataset.id_to_label)
                                val_writer.add_summary(summary, (val_step + 1) * config.Batch_size)
                                val_step += 1

                        IO_tool.save_hidden_state_collection()
                        IO_tool.hidden_states_statistics()
                        confusion_train_Lstm = np.zeros((number_of_classes, number_of_classes))
                        confusion_train_C3d = np.zeros((number_of_classes, number_of_classes))
                        confusion_train_Next = np.zeros((number_of_classes, number_of_classes))
                        confusion_val_Lstm = np.zeros((number_of_classes, number_of_classes))
                        confusion_val_C3d = np.zeros((number_of_classes, number_of_classes))
                        confusion_val_Next = np.zeros((number_of_classes, number_of_classes))
                        whole_saver.save(sess, config.model_filename, global_step=Network.global_step)


train()
