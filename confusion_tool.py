import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import config
import itertools
import tensorflow as tf
import textwrap
import re
import io
import itertools
import matplotlib


class confusion_tool:
    def __init__(self, number_of_classes, IO_tool, sess, Train_Net):
        self.number_of_classes = number_of_classes
        self.id_to_word = IO_tool.dataset.id_to_word
        self.Train_Net = Train_Net
        self.sess = sess
        word_list = [None] * self.number_of_classes
        for key in self.id_to_word:
            word_list[key] = self.id_to_word[key]
        self.word_list = word_list
        self.init_matrixes()
        self.word_list
        self.name_to_cm = {}
        self.name_to_cm['now_train'] = self.now_train
        self.name_to_cm['c3d_train'] = self.c3d_train
        self.name_to_cm['next_train'] = self.next_train
        self.name_to_cm['help_train'] = self.help_train
        self.name_to_cm['now_val'] = self.now_val
        self.name_to_cm['c3d_val'] = self.c3d_val
        self.name_to_cm['next_val'] = self.next_val
        self.name_to_cm['help_val'] = self.help_val
        self.name_to_summary = {}
        self.name_to_summary['now_train'] = Train_Net.now_train_confusion
        self.name_to_summary['c3d_train'] = Train_Net.c3d_train_confusion
        self.name_to_summary['next_train'] = Train_Net.next_train_confusion
        self.name_to_summary['help_train'] = Train_Net.help_train_confusion
        self.name_to_summary['now_val'] = Train_Net.now_val_confusion
        self.name_to_summary['c3d_val'] = Train_Net.c3d_val_confusion
        self.name_to_summary['next_val'] = Train_Net.next_val_confusion
        self.name_to_summary['help_val'] = Train_Net.help_val_confusion
        
    def init_matrixes(self):
        print('\n resetting confusions \n')
        self.now_train = np.zeros((self.number_of_classes, self.number_of_classes), dtype=np.int32)
        self.c3d_train= np.zeros((self.number_of_classes, self.number_of_classes), dtype=np.int32)
        self.next_train = np.zeros((self.number_of_classes, self.number_of_classes), dtype=np.int32)
        self.help_train = np.zeros((self.number_of_classes, self.number_of_classes), dtype=np.int32)
        self.now_val = np.zeros((self.number_of_classes, self.number_of_classes), dtype=np.int32)
        self.c3d_val= np.zeros((self.number_of_classes, self.number_of_classes), dtype=np.int32)
        self.next_val = np.zeros((self.number_of_classes, self.number_of_classes), dtype=np.int32)
        self.help_val = np.zeros((self.number_of_classes, self.number_of_classes), dtype=np.int32)

    def calculate_confusion_matrix(self, target, y_pred, tensor_name):
        cm = self.name_to_cm[tensor_name]
        shape_label = target.shape
        for i in range(shape_label[0]):
            true_label = target[i][0]
            actual_label = y_pred[i][0]
            cm[true_label, actual_label] += 1
        return cm

    def update_plot(self, cm_in, step, tensor_name, log_file):
        cm = np.zeros((self.number_of_classes, self.number_of_classes), dtype=np.float)
        for row in range(self.number_of_classes):
                sum_row = cm_in[row, :].sum()
                for col in range(self.number_of_classes):
                        place_sum = cm_in[row, col]
                        mean = 0
                        if place_sum > 0:
                                mean = float(place_sum)/float(sum_row)
                        cm[row, col] = mean
        fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, cmap='Oranges')
        
        tick_marks = np.arange(self.number_of_classes)

        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(self.word_list, fontsize=4, rotation=-90,  ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.word_list, fontsize=4, va ='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], '.2f') if cm[i,j]>0 else '.', horizontalalignment="center", fontsize=4, verticalalignment='center', color= "black")
            fig.set_tight_layout(True)

        if fig.canvas is None:
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        np_im = np.expand_dims(data, axis=0)
        summary = self.name_to_summary[tensor_name]
        summ = self.sess.run(summary, feed_dict={self.Train_Net.confusion_image: np_im})
        log_file.add_summary(summ, step)

    def update_confusion(self,data_collection, log_file, step):
        for entry in data_collection.keys():
            taget = data_collection[entry]['taget']
            y_pred = data_collection[entry]['y_pred']
            tensor_name = entry
            self.name_to_cm[tensor_name] = self.calculate_confusion_matrix(taget, y_pred, tensor_name)
            if step % config.update_confusion == 0:
                self.update_plot(self.name_to_cm[tensor_name], step, tensor_name, log_file)
                
        if step % config.reset_confusion_step == 0:
            self.init_matrixes()

    def save_confusion(self):
        if not os.path.exists('./results/confusion_matrix/' + folder_name + '/'):
            os.makedirs('./results/confusion_matrix/' + folder_name + '/')
        if (step) % 1000 == 0:
            plt.savefig('./results/confusion_matrix/' + folder_name + '/cm' + str(step) + '.png')
        else:
            plt.savefig('./results/confusion_matrix/' + folder_name + '/cm.png')