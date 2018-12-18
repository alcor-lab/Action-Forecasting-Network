import tensorflow as tf
import config


class Input_manager:
    def __init__(self, devices):

        with tf.name_scope('Inputs'):

            with tf.name_scope("Input"):
                self.input_batch = tf.placeholder(tf.float32, shape=(None, None, None, None, None, None), name="Input")
                self.h_input = tf.placeholder(tf.float32, shape=(None, None, None), name="Previous_hidden_state")
                self.c_input = tf.placeholder(tf.float32, shape=(None, None, None), name="Previous_hidden_state")

            with tf.name_scope("Target"):
                self.labels = tf.placeholder(tf.float32, shape=(None, None, None), name="Target")
                self.next_labels = tf.placeholder(tf.float32, shape=(None, None, None), name="next_labels")

            self.input_batch_list = []
            self.h_input_list = []
            self.c_input_list = []
            self.labels_list = []
            self.next_labels_list = []
            for j in range(devices):
                with tf.name_scope("Input_Net"):
                    with tf.name_scope("Input"):
                        self.input_batch_list.append(tf.squeeze(self.input_batch[j, :, :, :, :, :]))
                        squeezed_input_batch = self.input_batch
                        self.h_input_list.append(tf.squeeze(self.h_input[j, :, :]))
                        self.c_input_list.append(tf.squeeze(self.c_input[j, :, :]))

                    with tf.name_scope("Target"):
                        self.labels_list.append(tf.squeeze(self.labels[j, :, :]))
                        self.next_labels_list.append(tf.squeeze(self.next_labels[j, :, :]))

class activity_network:
    def __init__(self, number_of_classes, Input_manager, device_j):
        self.number_of_classes = number_of_classes
        self.next_fc = config.C3d_Output_features + number_of_classes

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
                wd_c = {
                    'wd_c1': self._variable_with_weight_decay('wd_c1', [8192, 2048], 0.04, 0.001),
                    'wd_c2': self._variable_with_weight_decay('wd_c2', [2048, config.pre_size], 0.04, 0.002),
                    'wd_cout': self._variable_with_weight_decay('wd_cout', [config.pre_size, self.number_of_classes], 0.04, 0.005)}
                bd_c = {
                    'bd_c1': self._variable_with_weight_decay('bd_c1', [2048], 0.04, 0.0),
                    'bd_c2': self._variable_with_weight_decay('bd_c2', [config.pre_size], 0.04, 0.0),
                    'bd_cout': self._variable_with_weight_decay('bd_cout', [self.number_of_classes], 0.04, 0.0)}
                wd_pL = {
                    'wd_pL1': self._variable_with_weight_decay('wd_pL1', [8192, 2048], 0.04, 0.001),
                    'wd_pL2': self._variable_with_weight_decay('wd_pL2', [2048, config.pre_size], 0.04, 0.002),
                    'wd_pLout': self._variable_with_weight_decay('wd_pLout', [config.pre_size, config.lstm_units], 0.04, 0.005)}
                bd_pL = {
                    'bd_pL1': self._variable_with_weight_decay('bd_pL1', [2048], 0.04, 0.0),
                    'bd_pL2': self._variable_with_weight_decay('bd_pL2', [config.pre_size], 0.04, 0.0),
                    'bd_pLout': self._variable_with_weight_decay('bd_pLout', [config.lstm_units], 0.04, 0.0)}
                wd_L = {
                    'wd_pLout_multiply': self._variable_with_weight_decay('wd_pLout_multiply', [3*config.lstm_units, config.lstm_units * config.lstm_units], 0.04, 0.005),
                    'wd_Lout': self._variable_with_weight_decay('wd_Lout', [config.lstm_units, self.number_of_classes], 0.04, 0.005),
                    'wd_Lout_next': self._variable_with_weight_decay('wd_Lout_next', [self.next_fc, self.number_of_classes], 0.04, 0.005)}
                bd_L = {
                    'bd_pLout_multiply': self._variable_with_weight_decay('bd_pLout_multiply', [config.lstm_units * config.lstm_units], 0.04, 0.0),
                    'bd_Lout': self._variable_with_weight_decay('bd_Lout', [self.number_of_classes], 0.04, 0.0),
                    'bd_Lout_next': self._variable_with_weight_decay('bd_Lout_next', [self.number_of_classes], 0.04, 0.0),
                    'bd_Lout_2': self._variable_with_weight_decay('bd_Lout_2', [config.lstm_units], 0.04, 0.0),
                    'bd_Lout_2_next': self._variable_with_weight_decay('bd_Lout_2_next', [config.lstm_units], 0.04, 0.0)}

            with tf.name_scope("Input"):
                self.input_batch = Input_manager.input_batch_list[device_j]
                # squeezed_input_batch = tf.squeeze(self.input_batch)
                squeezed_input_batch = self.input_batch
                self.h_input = Input_manager.h_input_list[device_j]
                self.c_input = Input_manager.c_input_list[device_j]

            with tf.name_scope("Target"):
                self.labels = Input_manager.labels_list[device_j]
                self.next_labels = Input_manager.next_labels_list[device_j]

            with tf.name_scope('C3d'):

                # Convolution Layer
                with tf.name_scope("Conv"):
                    conv1 = self.conv3d('conv1', squeezed_input_batch, wc['wc1'], bc['bc1'])
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
                    conv3 = self.conv3d('conv3b', conv3, wc['wc3b'], bc['bc3b'])
                    conv3 = tf.nn.leaky_relu(conv3, name='relu3b')
                    pool3 = self.max_pool('pool3', conv3, k=2)

                # Convolution Layer
                with tf.name_scope("Conv"):
                    conv4 = self.conv3d('conv4a', pool3, wc['wc4a'], bc['bc4a'])
                    conv4 = tf.nn.leaky_relu(conv4, name='relu4a')
                    conv4 = self.conv3d('conv4b', conv4, wc['wc4b'], bc['bc4b'])
                    conv4 = tf.nn.leaky_relu(conv4, name='relu4b')
                    pool4 = self.max_pool('pool4', conv4, k=2)

                # Convolution Layer
                with tf.name_scope("Conv"):
                    conv5 = self.conv3d('conv5a', pool4, wc['wc5a'], bc['bc5a'])
                    self.conv5 = tf.nn.leaky_relu(conv5, name='relu5a')
                    conv5 = self.conv3d('conv5b', conv5, wc['wc5b'], bc['bc5b'])
                    self.conv5 = tf.nn.leaky_relu(conv5, name='relu5b')
                    pool5 = self.max_pool('pool5', self.conv5, k=2)

                c3d_output = pool5

            with tf.name_scope("C3d_Fully_Connected"):
                # Fully connected layer
                reshape_1_cd = tf.transpose(c3d_output, perm=[0, 1, 4, 2, 3])
                reshape_2_cd = tf.contrib.layers.flatten(reshape_1_cd)

                dense1_cd = tf.nn.relu(tf.matmul(reshape_2_cd, wd_c['wd_c1']) + bd_c['bd_c1'], name='fc_c1')
                dense1_cd = tf.nn.dropout(dense1_cd, config.c3d_dropout)

                dense2_cd = tf.nn.relu(tf.matmul(dense1_cd, wd_c['wd_c2']) + bd_c['bd_c2'], name='fc_c2')
                dense2_cd = tf.nn.dropout(dense2_cd, config.c3d_dropout)

                self.out_cd = tf.matmul(dense2_cd, wd_c['wd_cout']) + bd_c['bd_cout']

            with tf.name_scope("C3d_classifier"):
                self.softmax_c3d = tf.nn.softmax(self.out_cd)
                self.predictions_c3d = tf.argmax(input=self.softmax_c3d, axis=1, name="c3d_prediction")

            with tf.name_scope("Pre_Lstm_Fully_Connected"):
                # Fully connected layer
                reshape_1_pL = tf.transpose(c3d_output, perm=[0, 1, 4, 2, 3])
                reshape_2_pL = tf.contrib.layers.flatten(reshape_1_pL)

                dense1_pL = tf.nn.relu(tf.matmul(reshape_2_pL, wd_pL['wd_pL1']) + bd_pL['bd_pL1'], name='fc_pL1')
                dense1_pL = tf.nn.dropout(dense1_pL, config.preLstm_dropout)

                dense2_pL = tf.nn.relu(tf.matmul(dense1_pL, wd_pL['wd_pL2']) + bd_pL['bd_pL2'], name='fc_pL2')
                dense2_pL = tf.nn.dropout(dense2_pL, config.preLstm_dropout)

                out_pL_0 = tf.matmul(dense2_pL, wd_pL['wd_pLout']) + bd_pL['bd_pLout']
                out_pL = tf.expand_dims(out_pL_0, 1)

            with tf.name_scope("Lstm"):
                encoder_cell = tf.contrib.rnn.LSTMCell(config.lstm_units)
                state = tf.contrib.rnn.LSTMStateTuple(self.c_input, self.h_input)
                decoder_output, decoder_state = tf.nn.dynamic_rnn(encoder_cell, out_pL,
                                                                  initial_state=state,
                                                                  dtype=tf.float32)
                decoder_output = tf.squeeze(decoder_output, axis=1)
                self.c_out = decoder_state.c
                self.h_out = decoder_state.h

            with tf.name_scope("Weight_Matrix_Generator"):
                composedVec = tf.concat([out_pL_0, self.c_input, self.h_input], axis=1)
                matrix_multiply = tf.matmul(composedVec, wd_L['wd_pLout_multiply']) + bd_L['bd_pLout_multiply']
                matrix_multiply = tf.reshape(matrix_multiply, [tf.shape(matrix_multiply)[0], config.lstm_units, config.lstm_units])

            with tf.name_scope("Current_Lstm_classifier"):
                out_L = tf.map_fn(lambda x: tf.squeeze(tf.matmul(tf.expand_dims(x[0], 0), x[1]) + bd_L['bd_Lout_2']), [decoder_output, matrix_multiply], dtype=tf.float32, parallel_iterations=config.Batch_size)
                self.out_L = tf.matmul(out_L, wd_L['wd_Lout']) + bd_L['bd_Lout']
                self.softmax_Lstm = tf.nn.softmax(self.out_L)
                self.predictions_Lstm = tf.argmax(input=self.softmax_Lstm, axis=1, name="classes")

            with tf.name_scope("Next_classifier"):
                out_L_next = tf.map_fn(lambda x: tf.squeeze(tf.matmul(tf.expand_dims(x[0], 0), x[1]) + bd_L['bd_Lout_2_next']), [decoder_output, matrix_multiply], dtype=tf.float32, parallel_iterations=config.Batch_size)
                out_L_next = composedVec = tf.concat([out_L_next, self.softmax_Lstm], axis=1)
                self.out_L_next = tf.matmul(out_L_next, wd_L['wd_Lout_next']) + bd_L['bd_Lout_next']
                self.softmax_Lstm_next = tf.nn.softmax(self.out_L_next)
                self.predictions_Lstm_next = tf.argmax(input=self.softmax_Lstm_next, axis=1, name="classes")

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


class Training:
    def __init__(self,Networks):
        self.cross_entropy_list = []
        self.accuracy_c3d_list = []
        self.accuracy_Lstm_list = []
        self.accuracy_Lstm_next_list = []
        self.cross_entropy_Lstm_list = []
        self.cross_entropy_Next_list = []
        self.cross_entropy_c3d_list = []
        with tf.name_scope('Losses_and_Metrics'):
            for Net in Networks:
                with tf.name_scope(Net):
                    with tf.name_scope('Metrics_calculation'):
                        casted_labels = tf.cast(Networks[Net].labels, tf.int64)
                        argmax_labels = tf.argmax(casted_labels, axis=1)
                        casted_labels_next = tf.cast(Networks[Net].next_labels, tf.int64)
                        argmax_labels_next = tf.argmax(casted_labels_next, axis=1)
                        accuracy_Lstm = tf.contrib.metrics.accuracy(Networks[Net].predictions_Lstm, argmax_labels)
                        accuracy_Lstm_next = tf.contrib.metrics.accuracy(Networks[Net].predictions_Lstm_next, argmax_labels_next)
                        accuracy_c3d = tf.contrib.metrics.accuracy(Networks[Net].predictions_c3d, argmax_labels)
                        comparison = tf.contrib.metrics.accuracy(argmax_labels_next, argmax_labels)

                    with tf.name_scope("C3d_Loss"):
                        cross_entropy_c3d_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].labels, logits=Networks[Net].out_cd)
                        cross_entropy_c3d = tf.reduce_sum(cross_entropy_c3d_vec)
                        cross_entropy_c3d = cross_entropy_c3d

                    with tf.name_scope("Lstm_Loss"):
                        cross_entropy_Lstm_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].labels, logits=Networks[Net].out_L)
                        cross_entropy_Lstm = tf.reduce_sum(cross_entropy_Lstm_vec)
                        cross_entropy_Lstm = cross_entropy_Lstm

                    with tf.name_scope("Next_Loss"):
                        cross_entropy_Next_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].next_labels, logits=Networks[Net].out_L_next)
                        cross_entropy_Next = tf.reduce_sum(cross_entropy_Next_vec)
                        cross_entropy_Next = cross_entropy_Next

                    with tf.name_scope("Global_Loss"):
                        cross_entropy = (accuracy_c3d)*((accuracy_Lstm)*cross_entropy_Next + (1 - accuracy_Lstm)*cross_entropy_Lstm) + (1 - accuracy_c3d) * cross_entropy_c3d

                    self.cross_entropy_list.append(cross_entropy)
                    self.accuracy_c3d_list.append(accuracy_c3d)
                    self.accuracy_Lstm_list.append(accuracy_Lstm)
                    self.accuracy_Lstm_next_list.append(accuracy_Lstm_next)
                    self.cross_entropy_Lstm_list.append(cross_entropy_Lstm)
                    self.cross_entropy_Next_list.append(cross_entropy_Next)
                    self.cross_entropy_c3d_list.append(cross_entropy_c3d)

        with tf.name_scope("Optimizer"):
            Train_variable = [v for v in Networks[Net].variables if 'Openpose' not in v.name.split('/')[0]]
            Train_variable = [v for v in Train_variable if 'MobilenetV1' not in v.name.split('/')[0]]
            starter_learning_rate = 0.0001
            # learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       # 1000, 0.9)
            tot_loss = sum(self.cross_entropy_list)

            self.train_op = tf.contrib.layers.optimize_loss(
                loss=tot_loss,
                global_step=Networks[Net].global_step,
                learning_rate=starter_learning_rate,
                optimizer='Adam',
                # clip_gradients=config.gradient_clipping_norm,
                variables=Train_variable)

        with tf.name_scope('Summary'):
            # tf.summary.histogram("c_out", self.c_out)
            # tf.summary.histogram("h_out", self.h_out)
            # tf.summary.histogram("c_in", self.c_input)
            # tf.summary.histogram("h_in", self.h_input)
            #
            # tf.summary.histogram("Lstm_classification", self.predictions_Lstm)
            # tf.summary.histogram("C3d_classification", self.predictions_c3d)
            # tf.summary.histogram("Next_prediction", self.predictions_Lstm_next)
            # tf.summary.histogram("labels_argmax", argmax_labels)
            # tf.summary.histogram("Next_argmax", argmax_labels_next)

            tf.summary.scalar('C3d_accuracy', sum(self.accuracy_c3d_list)/len(self.accuracy_c3d_list))
            tf.summary.scalar('Lstm_accuracy', sum(self.accuracy_Lstm_list)/len(self.accuracy_Lstm_list))
            tf.summary.scalar('Next_accuracy', sum(self.accuracy_Lstm_next_list)/len(self.accuracy_Lstm_next_list))
            tf.summary.scalar('Lstm_loss', sum(self.cross_entropy_Lstm_list)/len(self.cross_entropy_Lstm_list))
            tf.summary.scalar('Next_loss', sum(self.cross_entropy_Next_list)/len(self.cross_entropy_Next_list))
            tf.summary.scalar('C3d_Loss', sum(self.cross_entropy_c3d_list)/len(self.cross_entropy_c3d_list))

            self.merged = tf.summary.merge_all()

        with tf.name_scope('Outputs'):
            self.predictions_Lstm_list = []
            self.predictions_Lstm_next_list = []
            self.predictions_c3d_list = []
            self.c_out_list = []
            self.h_out_list = []
            for Net in Networks:
                self.predictions_Lstm_list.append(Networks[Net].predictions_Lstm)
                self.predictions_Lstm_next_list.append(Networks[Net].predictions_Lstm_next)
                self.predictions_c3d_list.append(Networks[Net].predictions_c3d)
                self.c_out_list.append(Networks[Net].c_out)
                self.h_out_list.append(Networks[Net].h_out)


        with tf.name_scope("Initializer"):
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            self.init = tf.group(init_local, init_global)
