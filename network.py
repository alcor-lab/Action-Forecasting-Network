import tensorflow as tf
import config


class Input_manager:
    def __init__(self, devices):

        with tf.name_scope('Inputs'):

            with tf.name_scope("Input"):
                self.input_batch = tf.placeholder(tf.float32, shape=(None, config.Batch_size, config.frames_per_step, config.out_H, config.out_W, config.input_channels), name="Input")
                self.h_input = tf.placeholder(tf.float32, shape=(None, config.Batch_size, config.lstm_units), name="Previous_hidden_state")
                self.c_input = tf.placeholder(tf.float32, shape=(None, config.Batch_size, config.lstm_units), name="Previous_hidden_state")

            with tf.name_scope("Target"):
                self.labels = tf.placeholder(tf.float32, shape=(None, None, None), name="Target")
                self.activity_labels = tf.placeholder(tf.float32, shape=(None, None, None), name="Target")
                self.next_labels = tf.placeholder(tf.float32, shape=(None, None, None), name="next_labels")
                self.multiple_next_labels = tf.placeholder(tf.float32, shape=(None, None, None), name="next_labels")
                self.history_labels = tf.placeholder(tf.float32, shape=(None, config.Batch_size, None, None), name="next_labels")


class activity_network:
    def __init__(self, number_of_classes, number_of_activities, max_history, Input_manager, device_j):
        self.number_of_classes = number_of_classes
        self.number_of_activities = number_of_activities
        # self.next_fc = config.C3d_Output_features + number_of_classes

        with tf.name_scope('Activity_Recognition_Network'):

            with tf.variable_scope('var_name'):
                wc = {
                    'wc1': self._variable_with_weight_decay('wc1', [3, 3, 3, config.input_channels, 64], 0.04, 0.00),
                    'wc2': self._variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
                    'wc3a': self._variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
                    'wc3b': self._variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
                    'wc4a': self._variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
                    'wc4b': self._variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
                    'wc5a': self._variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
                    'wc5b': self._variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00)}
                bc = {
                    'bc1': self._variable_with_weight_decay('bc1', [64], 0.04, 0.0),
                    'bc2': self._variable_with_weight_decay('bc2', [128], 0.04, 0.0),
                    'bc3a': self._variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
                    'bc3b': self._variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
                    'bc4a': self._variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
                    'bc4b': self._variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
                    'bc5a': self._variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
                    'bc5b': self._variable_with_weight_decay('bc5b', [512], 0.04, 0.0)}
                # wd_c = {
                #     'wd_c1': self._variable_with_weight_decay('wd_c1', [8192, 2048], 0.04, 0.001),
                #     'wd_c2': self._variable_with_weight_decay('wd_c2', [2048, config.pre_size], 0.04, 0.002),
                #     'wd_cout': self._variable_with_weight_decay('wd_cout', [config.pre_size, self.number_of_classes], 0.04, 0.005)}
                # bd_c = {
                #     'bd_c1': self._variable_with_weight_decay('bd_c1', [2048], 0.04, 0.0),
                #     'bd_c2': self._variable_with_weight_decay('bd_c2', [config.pre_size], 0.04, 0.0),
                #     'bd_cout': self._variable_with_weight_decay('bd_cout', [self.number_of_classes], 0.04, 0.0)}
                # wd_pL = {
                #     'wd_pL1': self._variable_with_weight_decay('wd_pL1', [8192, 2048], 0.04, 0.001),
                #     'wd_pL2': self._variable_with_weight_decay('wd_pL2', [2048, config.pre_size], 0.04, 0.002),
                #     'wd_pLout': self._variable_with_weight_decay('wd_pLout', [config.pre_size, config.lstm_units], 0.04, 0.005)}
                # bd_pL = {
                #     'bd_pL1': self._variable_with_weight_decay('bd_pL1', [2048], 0.04, 0.0),
                #     'bd_pL2': self._variable_with_weight_decay('bd_pL2', [config.pre_size], 0.04, 0.0),
                #     'bd_pLout': self._variable_with_weight_decay('bd_pLout', [config.lstm_units], 0.04, 0.0)}
                # wd_L = {
                #     'wd_pLout_multiply': self._variable_with_weight_decay('wd_pLout_multiply', [3*config.lstm_units, config.lstm_units * config.lstm_units], 0.04, 0.005),
                #     'wd_Lout': self._variable_with_weight_decay('wd_Lout', [config.lstm_units, self.number_of_classes], 0.04, 0.005),
                #     'wd_Lout_next': self._variable_with_weight_decay('wd_Lout_next', [self.next_fc, self.number_of_classes], 0.04, 0.005),
                #     'wd_Lout_multiple_next': self._variable_with_weight_decay('wd_Lout_multiple_next', [self.next_fc, self.number_of_classes], 0.04, 0.005)}
                # bd_L = {
                #     'bd_pLout_multiply': self._variable_with_weight_decay('bd_pLout_multiply', [config.lstm_units * config.lstm_units], 0.04, 0.0),
                #     'bd_Lout': self._variable_with_weight_decay('bd_Lout', [self.number_of_classes], 0.04, 0.0),
                #     'bd_Lout_next': self._variable_with_weight_decay('bd_Lout_next', [self.number_of_classes], 0.04, 0.0),
                #     'bd_Lout_multiple_next': self._variable_with_weight_decay('bd_Lout_multiple_next', [self.number_of_classes], 0.04, 0.0),
                #     'bd_attention': self._variable_with_weight_decay('bd_Lout_2', [config.lstm_units], 0.04, 0.0),
                #     'bd_Lout_2_next': self._variable_with_weight_decay('bd_Lout_2_next', [config.lstm_units], 0.04, 0.0),
                #     'bd_Lout_2_multiple_next': self._variable_with_weight_decay('bd_Lout__multiple2_next', [config.lstm_units], 0.04, 0.0)}

            with tf.name_scope("Input"):
                self.input_batch = tf.squeeze(Input_manager.input_batch[device_j, :, :, :, :, :])
                self.h_input = tf.squeeze(Input_manager.h_input[device_j, :, :])
                self.c_input = tf.squeeze(Input_manager.c_input[device_j, :, :])

            with tf.name_scope("Target"):
                self.labels = tf.squeeze(Input_manager.labels[device_j, :, :])
                # self.activity_labels = tf.squeeze(Input_manager.activity_labels[device_j, :, :])
                self.activity_labels = Input_manager.activity_labels[device_j, :, :]
                # print(Input_manager.activity_labels)
                # print(Input_manager.activity_labels[device_j, :, :])
                # print(self.activity_labels)
                self.next_labels = tf.squeeze(Input_manager.next_labels[device_j, :, :])
                self.multiple_next_labels = tf.squeeze(Input_manager.multiple_next_labels[device_j, :, :])
                self.history_labels = tf.squeeze(Input_manager.history_labels[device_j, :, :, :])

            with tf.name_scope('C3d'):

                # Convolution Layer
                with tf.name_scope("Conv"):
                    conv1 = self.conv3d('conv1', self.input_batch, wc['wc1'], bc['bc1'])
                    conv1 = tf.nn.leaky_relu(conv1, name='relu1')
                    pool1 = self.max_pool('pool1', conv1, k=1)

                # Convolution Layer
                with tf.name_scope("Conv"):
                    conv2 = self.conv3d('conv2', pool1, wc['wc2'], bc['bc2'])
                    conv2 = tf.nn.leaky_relu(conv2, name='relu2')
                    pool2 = self.max_pool('pool2', conv2, k=2)

                # Convolution Layer
                with tf.name_scope("Conv"):
                    conv3 = self.conv3d('conv3a', pool2, wc['wc3a'], bc['bc3a'])
                    conv3 = tf.nn.leaky_relu(conv3, name='relu3a')
                    # conv3 = self.conv3d('conv3b', conv3, wc['wc3b'], bc['bc3b'])
                    # conv3 = tf.nn.leaky_relu(conv3, name='relu3b')
                    pool3 = self.max_pool('pool3', conv3, k=2)

                # Convolution Layer
                with tf.name_scope("Conv"):
                    conv4 = self.conv3d('conv4a', pool3, wc['wc4a'], bc['bc4a'])
                    conv4 = tf.nn.leaky_relu(conv4, name='relu4a')
                    # conv4 = self.conv3d('conv4b', conv4, wc['wc4b'], bc['bc4b'])
                    # conv4 = tf.nn.leaky_relu(conv4, name='relu4b')
                    pool4 = self.max_pool('pool4', conv4, k=2)

                # Convolution Layer
                with tf.name_scope("Conv"):
                    conv5 = self.conv3d('conv5a', pool4, wc['wc5a'], bc['bc5a'])
                    self.conv5 = tf.nn.leaky_relu(conv5, name='relu5a')
                    # conv5 = self.conv3d('conv5b', conv5, wc['sc5b'], bc['bc5b'])
                    # self.conv5 = tf.nn.leaky_relu(conv5, name=srelu5b')
                    pool5 = self.max_pool('pool5', self.conv5, k=2)

                c3d_output = pool5

            with tf.name_scope('reshape_c3d'):
                reshape_1_cd = tf.transpose(c3d_output, perm=[0, 1, 4, 2, 3])
                self.reshape_2_cd = tf.contrib.layers.flatten(reshape_1_cd)

            with tf.name_scope("Encoder"):
                dense1_cd = tf.layers.dense(self.reshape_2_cd, config.enc_fc_1)
                dense2_cd = tf.layers.dense(dense1_cd, config.enc_fc_2)
                self.out_pL = tf.layers.dense(dense2_cd, config.lstm_units)
                exp_out_pL = tf.expand_dims(self.out_pL, 1)

            with tf.name_scope("Decoder"):
                # Fully connected layer
                dense1_cd = tf.layers.dense(self.out_pL, config.enc_fc_2)
                dense2_cd = tf.layers.dense(dense1_cd, config.enc_fc_1)
                self.autoenc_out = tf.layers.dense(dense2_cd, self.reshape_2_cd.shape[-1])

            with tf.name_scope("Lstm"):
                encoder_cell = tf.contrib.rnn.LSTMCell(config.lstm_units, name='now_cell')
                state = tf.contrib.rnn.LSTMStateTuple(self.c_input, self.h_input)
                decoder_output, decoder_state = tf.nn.dynamic_rnn(encoder_cell, exp_out_pL,
                                                                  initial_state=state,
                                                                  dtype=tf.float32)
                history_input = decoder_output
                decoder_output = tf.squeeze(decoder_output, axis=1)
                self.c_out = decoder_state.c
                self.h_out = decoder_state.h

            with tf.name_scope('history_decoder'):
                shape_H = history_input.shape
                history_input = tf.tile(history_input, [1, max_history, 1])
                history_decoder_cell = tf.contrib.rnn.LSTMCell(config.lstm_units, name='history_cell')
                History_output, history_state = tf.nn.dynamic_rnn(history_decoder_cell, history_input,
                                                                  dtype=tf.float32)

            with tf.name_scope("Activity_Classifier"):
                self.activity_logit = tf.layers.dense(decoder_output, config.pre_class)
                self.activity_logit = tf.layers.dense(self.activity_logit, self.number_of_activities)
                self.activity_softmax = tf.nn.softmax(self.activity_logit)
                self.activity_predictions = tf.argmax(input=self.activity_softmax, axis=1, name="classes")
                self.activity_one_hot_prediction= tf.one_hot(self.activity_predictions, depth = self.activity_softmax.shape[-1])

            with tf.name_scope("c3d_classifier"):
                composedVec = tf.concat([self.activity_logit, self.reshape_2_cd], axis=1)
                self.out_cd = tf.layers.dense(composedVec, config.pre_class)
                self.out_cd = tf.layers.dense(self.out_cd, self.number_of_classes)
                self.softmax_c3d = tf.nn.softmax(self.out_cd)
                self.predictions_c3d = tf.argmax(input=self.softmax_c3d, axis=1, name="c3d_prediction")
                self.c3d_one_hot_prediction= tf.one_hot(self.predictions_c3d, depth = self.softmax_c3d.shape[-1])

            with tf.name_scope('history_classifier'):
                self.history_logit = tf.layers.dense(History_output, config.pre_class)
                self.history_logit = tf.layers.dense(self.history_logit, self.number_of_classes)
                self.history_softmax = tf.nn.softmax(self.history_logit)
                self.history_predictions = tf.argmax(input=self.history_softmax, axis=2, name="classes")
                self.history_one_hot_prediction= tf.one_hot(self.history_predictions, depth = self.history_softmax.shape[-1])
                casted_history = tf.cast(self.history_predictions, tf.float32)

            with tf.name_scope("Attention_Generator"):
                if config.matrix_attention:
                    composedVec = tf.concat([self.activity_logit, self.c_input, self.h_input, self.out_cd, casted_history], axis=1)
                    matrix_multiply = tf.sigmoid(tf.layers.dense(composedVec, config.lstm_units))
                    attention_out = tf.multiply(decoder_output, matrix_multiply)
                else:
                    composedVec = tf.concat([decoder_output, self.activity_logit, self.c_input, self.h_input, self.out_cd, casted_history], axis=1)
                    attention_out = tf.layers.dense(composedVec, config.lstm_units)

            with tf.name_scope("Now_Classifier"):
                input_now= tf.concat([attention_out, self.softmax_c3d, casted_history], axis=1)
                self.out_L = tf.layers.dense(input_now, config.pre_class)
                self.out_L = tf.layers.dense(self.out_L, self.number_of_classes)
                self.softmax_Lstm = tf.nn.softmax(self.out_L)
                self.predictions_Lstm = tf.argmax(input=self.softmax_Lstm, axis=1, name="classes")
                self.now_one_hot_prediction= tf.one_hot(self.predictions_Lstm, depth = self.softmax_Lstm.shape[-1])

            with tf.name_scope("Multiple_Next_classifier"):
                input_multiple_next = tf.concat([attention_out, self.softmax_c3d, self.softmax_Lstm, casted_history], axis=1)
                self.out_L_multiple_next = tf.layers.dense(input_multiple_next, config.pre_class)
                self.out_L_multiple_next = tf.layers.dense(self.out_L_multiple_next, self.number_of_classes)
                self.sigmoid_multiple_next = tf.sigmoid(self.out_L_multiple_next)
                self.multiple_one_hot_prediction = tf.to_int32(self.sigmoid_multiple_next > 0.5)
                self.output_multiple_next_label = tf.where(tf.greater(self.multiple_one_hot_prediction, 0))[:,1]

            with tf.name_scope("Next_classifier"):
                in_next = tf.concat([attention_out, self.softmax_c3d, self.softmax_Lstm, self.sigmoid_multiple_next, casted_history], axis=1)
                self.out_L_next = tf.layers.dense(in_next, config.pre_class)
                self.out_L_next = tf.layers.dense(self.out_L_next, self.number_of_classes)
                self.softmax_Lstm_next = tf.nn.softmax(self.out_L_next)
                self.predictions_Lstm_next = tf.argmax(input=self.softmax_Lstm_next, axis=1, name="classes")
                self.next_one_hot_prediction = tf.one_hot(self.predictions_Lstm_next, depth = self.softmax_Lstm_next.shape[-1])

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.nn.l2_loss(var) * wd
            tf.add_to_collection('losses', weight_decay)
        return var

    def conv3d(self, name, l_input, w, b):
        return tf.nn.bias_add(
            tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
            b)

    def max_pool(self, name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

    # def fc_network(input, number_of_layers, layers_dimension, activation='None', dropout='None'):
    #     if not isinstance(activation, (list,)):
    #         activation = [activation]
    #         for i in range(1, number_of_layers)
    #             activation.append(activation[1])
    #     else:
    #         if len(activation) != number_of_layers:
    #             Error_string = 'Number of activation is different from number of layers/
    #                            'number of activation given:' + str(len(activation)) +
    #                            'while number of layer is' + str(number_of_layers)
    #             raise ValueError(Error_string)
    #
    #     if not isinstance(activation, (list,)):
    #         dropout = [dropout]
    #         for i in range(1, number_of_layers)
    #             activation.append(dropout[1])
    #     else
    #         if len(dropout) != number_of_layers:
    #             Error_string = 'Number of dropout is different from number of layers/
    #                            'number of dropout given:' + str(len(dropout)) +
    #                            'while number of layer is' + str(number_of_layers)
    #             raise ValueError(Error_string)
    #
    #     for j in number_of_layers:
    #         if j = 0:
    #             fc_layer = tf.layers.dense(input_now, layers_dimension[j], name = 'fc' + str(j), activation=activation[j])
    #             if dropout[j] != None:
    #                 dropout = tf.nn.dropout(fc_layer, dropout[j])
    #         else:
    #             fc_layer = tf.layers.dense(fc_layer, layers_dimension[j], name = 'fc' + str(j, activation=activation[j])
    #             if dropout[j] != None:
    #                 dropout = tf.nn.dropout(fc_layer, dropout[j])
    #     return fc_layer


class Training:
    def __init__(self,Networks):
        # self.c3d_accuracy_list = []
        # self.now_accuracy_list = []
        # self.multi_accuracy_list = []
        # self.next_accuracy_list = []
        # self.c3d_precision_list = []
        # self.now_precision_list = []
        # self.multi_precision_list = []
        # self.next_precision_list = []
        # self.c3d_recall_list = []
        # self.now_recall_list = []
        # self.multi_recall_list = []
        # self.next_recall_list = []
        # self.c3d_f1_list = []
        # self.now_f1_list = []
        # self.multi_f1_list = []
        # self.next_f1_list = []
        # self.cross_entropy_list = []
        # self.cross_entropy_Lstm_list = []
        # self.cross_entropy_Next_list = []
        # self.cross_entropy_multiple_Next_list = []
        # self.cross_entropy_c3d_list = []

        with tf.name_scope('Loaders_and_Savers'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.variables = tf.contrib.slim.get_variables_to_restore()
            with tf.name_scope('Model_Saver'):
                self.model_saver = tf.train.Saver()
            with tf.name_scope('C3D_Loaders'):
                Load = ['wc2', 'wc3a', 'wc3b', 'wc4a', 'wc4b', 'wc5a', 'wc5b', 'bc1',
                        'bc2', 'bc3a', 'bc3b', 'bc4a', 'bc4b', 'bc5a', 'bc5b']
                c3d_loader_variables = [v for v in self.variables if 'Network/var_name' in v.name.split(':')[0] and v.name.split(':')[0].split('/')[-1] in Load]
                name_to_vars = {''.join(v.op.name.split('Network/')[0:]): v for v in c3d_loader_variables}
                self.c3d_loader = tf.train.Saver(name_to_vars)

        with tf.name_scope('Metrics'):
            z = 0
            for Net in Networks:
                if z == 0:
                    c3d_pred_conc = Networks[Net].c3d_one_hot_prediction
                    activity_pred_conc = Networks[Net].activity_one_hot_prediction
                    now_pred_conc = Networks[Net].now_one_hot_prediction
                    multi_pred_conc = Networks[Net].multiple_one_hot_prediction
                    next_pred_conc = Networks[Net].next_one_hot_prediction
                    history_pred_conc = Networks[Net].history_one_hot_prediction
                    activity_label_conc = Networks[Net].activity_labels
                    now_label_conc = Networks[Net].labels
                    multi_label_conc = Networks[Net].multiple_next_labels
                    next_label_conc = Networks[Net].next_labels
                    history_label_conc = Networks[Net].history_labels
                    z +=1
                else:
                    c3d_pred_conc = tf.concat([c3d_pred_conc, Networks[Net].c3d_one_hot_prediction], axis=0)
                    activity_pred_conc = tf.concat([activity_pred_conc, Networks[Net].activity_one_hot_prediction], axis=0)
                    now_pred_conc = tf.concat([now_pred_conc, Networks[Net].now_one_hot_prediction], axis=0)
                    multi_pred_conc = tf.concat([multi_pred_conc, Networks[Net].multiple_one_hot_prediction], axis=0)
                    next_pred_conc = tf.concat([next_pred_conc, Networks[Net].next_one_hot_prediction], axis=0)
                    history_pred_conc = tf.concat([history_pred_conc, Networks[Net].history_one_hot_prediction], axis=0)
                    activity_label_conc = tf.concat([activity_label_conc, Networks[Net].activity_labels], axis=0)
                    now_label_conc = tf.concat([now_label_conc, Networks[Net].labels], axis=0)
                    multi_label_conc = tf.concat([multi_label_conc, Networks[Net].multiple_next_labels], axis=0)
                    next_label_conc = tf.concat([next_label_conc, Networks[Net].next_labels], axis=0)
                    history_label_conc = tf.concat([history_label_conc, Networks[Net].history_labels], axis=0)

            with tf.name_scope('Metrics_calculation'):
                c3d_precision, c3d_recall, c3d_f1, c3d_accuracy = self.accuracy_metrics(c3d_pred_conc, now_label_conc)
                activity_precision, activity_recall, activity_f1, activity_accuracy = self.accuracy_metrics(activity_pred_conc, activity_label_conc)
                now_precision, now_recall, now_f1, now_accuracy = self.accuracy_metrics(now_pred_conc, now_label_conc)
                multi_precision, multi_recall, multi_f1, multi_accuracy = self.accuracy_metrics(multi_pred_conc, multi_label_conc)
                next_precision, next_recall, next_f1, next_accuracy = self.accuracy_metrics(next_pred_conc, next_label_conc)
                history_precision, history_recall, history_f1, history_accuracy = self.accuracy_metrics(history_pred_conc, history_label_conc)

        with tf.name_scope('Loss'):
            for Net in Networks:
                z = 0
                with tf.name_scope(Net):
                    with tf.name_scope("C3d_Loss"):
                        cross_entropy_c3d_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].labels, logits=Networks[Net].out_cd)
                        cross_entropy_c3d = tf.reduce_sum(cross_entropy_c3d_vec)

                    with tf.name_scope("activity_Loss"):
                        cross_entropy_activity = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].activity_labels, logits=Networks[Net].activity_logit)
                        cross_entropy_activity = tf.reduce_sum(cross_entropy_activity)

                    with tf.name_scope("Lstm_Loss"):
                        cross_entropy_Lstm_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].labels, logits=Networks[Net].out_L)
                        cross_entropy_Lstm = tf.reduce_sum(cross_entropy_Lstm_vec)

                    with tf.name_scope("Next_Loss"):
                        cross_entropy_Next_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].next_labels, logits=Networks[Net].out_L_next)
                        cross_entropy_Next = tf.reduce_sum(cross_entropy_Next_vec)

                    with tf.name_scope("Multiple_Next_Loss"):
                        cross_entropy_multiple_Next_vec = tf.nn.weighted_cross_entropy_with_logits(targets=Networks[Net].multiple_next_labels, logits=Networks[Net].out_L_multiple_next, pos_weight=10)
                        cross_entropy_multiple_Next = tf.reduce_sum(cross_entropy_multiple_Next_vec)

                    with tf.name_scope("History_Loss"):
                        cross_entropy_history_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].history_labels, logits=Networks[Net].history_logit)
                        cross_entropy_history = tf.reduce_sum(cross_entropy_history_vec)

                    with tf.name_scope("Autoencoder_Loss"):
                        auto_enc_loss=tf.reduce_mean(tf.square(Networks[Net].autoenc_out-Networks[Net].reshape_2_cd))

                    if z == 0:
                        next_loss_sum = cross_entropy_Next
                        c3d_loss_sum = cross_entropy_c3d
                        activity_loss_sum = cross_entropy_activity
                        now_loss_sum = cross_entropy_Lstm
                        multi_loss_sum = cross_entropy_multiple_Next
                        auto_enc_loss_sum = auto_enc_loss
                        history_loss_sum = cross_entropy_history
                        z += 1
                    else:
                        next_loss_sum += cross_entropy_Next
                        c3d_loss_sum += cross_entropy_c3d
                        activity_loss_sum += cross_entropy_activity
                        now_loss_sum += cross_entropy_Lstm
                        multi_loss_sum += cross_entropy_multiple_Next
                        auto_enc_loss_sum += auto_enc_loss
                        history_loss_sum += cross_entropy_history

            with tf.name_scope("Global_Loss"):
                # cross_entropy = (accuracy_c3d)*((accuracy_Lstm)*(cross_entropy_Next + cross_entropy_multiple_Next) + (1 - accuracy_Lstm)*cross_entropy_Lstm) + (1 - accuracy_c3d) * cross_entropy_c3d
                # cross_entropy = (c3d_accuracy)*(cross_entropy_Next + cross_entropy_multiple_Next+ cross_entropy_Lstm) + (1 - c3d_accuracy) * cross_entropy_c3d
                # cross_entropy = cross_entropy_Next + cross_entropy_multiple_Next + cross_entropy_Lstm + cross_entropy_c3d
                # print(c3d_accuracy)
                # print(now_accuracy)
                # print(multi_f1)
                # print(next_loss_sum)
                # print(next_loss_sum)
                # print(multi_loss_sum)
                # print(c3d_loss_sum)
                c3d_loss_sum = tf.cast(c3d_loss_sum, tf.float64)
                activity_loss_sum = tf.cast(c3d_loss_sum, tf.float64)
                multi_loss_sum = tf.cast(multi_loss_sum, tf.float64)
                next_loss_sum = tf.cast(next_loss_sum, tf.float64)
                now_loss_sum = tf.cast(now_loss_sum, tf.float64)
                auto_enc_loss_sum = tf.cast(auto_enc_loss_sum, tf.float64)
                history_loss_sum = tf.cast(history_loss_sum, tf.float64)
                total_loss = (c3d_recall)*(activity_recall*(now_recall * (multi_f1 * next_loss_sum + (1-multi_f1) * multi_loss_sum + history_loss_sum) + (1-now_recall)*now_loss_sum) + (1 - activity_recall)*activity_loss_sum) + (1 - c3d_recall) * c3d_loss_sum + auto_enc_loss_sum

        with tf.name_scope("Optimizer"):
            Train_variable = [v for v in self.variables if 'Openpose' not in v.name.split('/')[0]]
            Train_variable = [v for v in Train_variable if 'MobilenetV1' not in v.name.split('/')[0]]

            starter_learning_rate = 0.0001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       1000, 0.9)
            self.train_op = tf.contrib.layers.optimize_loss(
                loss=total_loss,
                global_step=self.global_step,
                learning_rate=learning_rate,
                optimizer='Adam',
                clip_gradients=config.gradient_clipping_norm,
                variables=Train_variable)

        with tf.name_scope('Summary'):
            # tf.summary.histogram("c_out", self.c_out)
            # tf.summary.histogram("h_out", self.h_out)
            # tf.summary.histogram("c_in", self.c_input)
            # tf.summary.histogram("h_in", self.h_input)
            j = 0
            for Net in Networks:
                if j == 0:
                    conc_Lstm_classification = Networks[Net].predictions_Lstm
                    conc_predictions_c3d = Networks[Net].predictions_c3d
                    conc_predictions_Lstm_next = Networks[Net].predictions_Lstm_next
                    conc_output_multiple_next_label = Networks[Net].output_multiple_next_label
                    conc_predictions_Lstm = Networks[Net].predictions_Lstm
                    conc_labels = Networks[Net].labels
                    conc_next_labels = Networks[Net].next_labels
                    conc_multiple_next_labels = Networks[Net].multiple_next_labels
                    conc_activity_labels = Networks[Net].activity_labels
                else:
                    conc_Lstm_classification = tf.concat([conc_Lstm_classification,Networks[Net].predictions_Lstm], axis=0)
                    conc_predictions_c3d = tf.concat([conc_predictions_c3d,Networks[Net].predictions_c3d], axis=0)
                    conc_predictions_Lstm_next = tf.concat([conc_predictions_Lstm_next,Networks[Net].predictions_Lstm_next], axis=0)
                    conc_output_multiple_next_label = tf.concat([conc_output_multiple_next_label,Networks[Net].output_multiple_next_label], axis=0)
                    conc_predictions_Lstm = tf.concat([conc_predictions_Lstm,Networks[Net].predictions_Lstm], axis=0)
                    conc_labels = tf.concat([conc_labels,Networks[Net].labels], axis=0)
                    conc_next_labels = tf.concat([conc_next_labels,Networks[Net].next_labels], axis=0)
                    conc_multiple_next_labels = tf.concat([conc_multiple_next_labels,Networks[Net].multiple_next_labels], axis=0)
                    conc_activity_labels = tf.concat([conc_activity_labels,Networks[Net].activity_labels], axis=0)
                j += 1

            tf.summary.histogram("Lstm_classification", conc_predictions_Lstm)
            tf.summary.histogram("activity_classification", conc_predictions_c3d)
            tf.summary.histogram("Next_prediction", conc_predictions_Lstm_next)
            tf.summary.histogram("Multiple_Next_prediction", conc_output_multiple_next_label)
            casted_labels = tf.cast(conc_labels, tf.int64)
            argmax_labels = tf.argmax(casted_labels, axis=1)
            casted_activity_labels = tf.cast(conc_activity_labels, tf.int64)
            argmax_activity_labels = tf.argmax(casted_activity_labels, axis=1)
            casted_labels_next = tf.cast(conc_next_labels, tf.int64)
            argmax_labels_next = tf.argmax(casted_labels_next, axis=1)
            check_value = tf.constant([1], tf.int32)
            casted_multiple_next_labels = tf.cast(conc_multiple_next_labels, tf.int32)
            label_multiple_next = tf.where(tf.greater(casted_multiple_next_labels, 0))[:,1]
            tf.summary.histogram("labels_target", argmax_labels)
            tf.summary.histogram("activity_target", argmax_activity_labels)
            tf.summary.histogram("multiple_next_target", label_multiple_next)
            tf.summary.histogram("Next_target", argmax_labels_next)
            # tf.summary.scalar('c3d_accuracy', c3d_accuracy)
            # tf.summary.scalar('c3d_precision', sum(self.c3d_precision_list)/len(self.c3d_precision_list))
            tf.summary.scalar('c3d_recall', c3d_recall)
            # tf.summary.scalar('c3d_f1', sum(self.c3d_f1_list)/len(self.c3d_f1_list))

            # tf.summary.scalar('now_accuracy', now_accuracy)
            # tf.summary.scalar('now_precision', sum(self.now_precision_list)/len(self.now_precision_list))
            tf.summary.scalar('now_recall', now_recall)
            # tf.summary.scalar('now_f1', sum(self.now_f1_list)/len(self.now_f1_list))

            tf.summary.scalar('multi_accuracy', multi_accuracy)
            tf.summary.scalar('multi_precision', multi_precision)
            tf.summary.scalar('multi_recall', multi_recall)
            tf.summary.scalar('multi_f1', multi_f1)

            # tf.summary.scalar('next_accuracy', next_accuracy)
            # tf.summary.scalar('next_precision', sum(self.next_precision_list)/len(self.next_precision_list))
            tf.summary.scalar('next_recall', next_recall)
            # tf.summary.scalar('next_f1', sum(self.next_f1_list)/len(self.next_f1_list))

            tf.summary.scalar('history_recall', history_recall)
            tf.summary.scalar('history_accuracy', history_accuracy)
            tf.summary.scalar('history_precision', history_precision)
            tf.summary.scalar('history_f1', history_f1)
            # tf.summary.scalar('C3d_accuracy', sum(self.accuracy_c3d_list)/len(self.accuracy_c3d_list))
            # tf.summary.scalar('Lstm_accuracy', sum(self.accuracy_Lstm_list)/len(self.accuracy_Lstm_list))
            # tf.summary.scalar('Next_accuracy', sum(self.accuracy_Lstm_next_list)/len(self.accuracy_Lstm_next_list))
            # tf.summary.scalar('Multiple_Next_accuracy', sum(self.accuracy_Lstm_multiple_next_list)/len(self.accuracy_Lstm_multiple_next_list))
            tf.summary.scalar('Lstm_loss', now_loss_sum)
            tf.summary.scalar('Next_loss', next_loss_sum)
            tf.summary.scalar('multiple_Next_loss', multi_loss_sum)
            tf.summary.scalar('history_loss', history_loss_sum)
            tf.summary.scalar('auto_enc_loss_sum', auto_enc_loss_sum)
            tf.summary.scalar('C3d_Loss', c3d_loss_sum)

            self.merged = tf.summary.merge_all()

        with tf.name_scope('Outputs'):
            self.predictions_Lstm_list = []
            self.predictions_Lstm_next_list = []
            self.predictions_c3d_list = []
            self.c_out_list = []
            self.h_out_list = []
            self.multi_sigmoid_list = []
            self.forecast_softmax_list = []
            for Net in Networks:
                self.predictions_Lstm_list.append(Networks[Net].predictions_Lstm)
                self.predictions_Lstm_next_list.append(Networks[Net].predictions_Lstm_next)
                self.predictions_c3d_list.append(Networks[Net].predictions_c3d)
                self.c_out_list.append(Networks[Net].c_out)
                self.h_out_list.append(Networks[Net].h_out)
                self.multi_sigmoid_list.append(Networks[Net].sigmoid_multiple_next)
                self.forecast_softmax_list.append(Networks[Net].softmax_Lstm_next)

        with tf.name_scope("Initializer"):
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            self.init = tf.group(init_local, init_global)


    def accuracy_metrics(self, predicted, actual):
        with tf.name_scope('accuracy_metrics'):
            predicted = tf.cast(predicted, tf.int64)
            actual = tf.cast(actual, tf.int64)
            TP = tf.count_nonzero(predicted * actual)
            TN = tf.count_nonzero((predicted - 1) * (actual - 1))
            FP = tf.count_nonzero(predicted * (actual - 1))
            FN = tf.count_nonzero((predicted - 1) * actual)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            # accuracy = (TP) / (TP + FP + TN + FN)
            # accuracy = (TP) / (TP + FP + TN + FN)
            accuracy = (TP + TN) / (TP + FP + TN + FN)
        return precision, recall, f1, accuracy
