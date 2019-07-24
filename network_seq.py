import tensorflow as tf
import config
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pprint
pp = pprint.PrettyPrinter(indent=4)



class Input_manager:
    def __init__(self, devices, IO_tool):

        with tf.name_scope("Input"):
            self.input_batch = tf.placeholder(tf.uint8, shape=(None, None, config.seq_len, config.frames_per_step, config.out_H, config.out_W, config.input_channels), name="Input")
            self.h_input = tf.placeholder(tf.float32, shape=(None, len(config.encoder_lstm_layers), None, config.lstm_units), name="h_input")
            self.c_input = tf.placeholder(tf.float32, shape=(None, len(config.encoder_lstm_layers), None, config.lstm_units), name="c_input")
            self.c_output = self.c_input
            self.h_output = self.h_input
            self.drop_out_prob = tf.placeholder_with_default(1.0, shape=())
            self.in_lstm_drop_out_prob = tf.placeholder_with_default(1.0, shape=())
            self.out_lstm_drop_out_prob = tf.placeholder_with_default(1.0, shape=())
            self.state_lstm_drop_out_prob = tf.placeholder_with_default(1.0, shape=())

        with tf.name_scope('Object_Input'):
            self.obj_input = tf.placeholder(tf.float32, shape=(None, None, config.seq_len, len(IO_tool.dataset.word_to_id)), name="obj_input")
        
        with tf.name_scope("Target"):
            self.labels = tf.placeholder(tf.int32, shape=(None, None, config.seq_len + 1), name="now_label")
            self.help_labels = tf.placeholder(tf.int32, shape=(None, None, 4), name="help_label")
            self.next_labels = tf.placeholder(tf.int32, shape=(None, None), name="next_label")
            self.dec_embeddings = tf.Variable(tf.random_uniform([len(IO_tool.dataset.word_to_id), config.decoder_embedding_size]))

class activity_network:
    def __init__(self, number_of_classes, Input_manager, device_j, IO_tool):
        self.number_of_classes = number_of_classes
        self.out_vocab_size = len(IO_tool.dataset.word_to_id)

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

            with tf.name_scope("Input"):
                self.input_batch = Input_manager.input_batch[device_j, :, :, :, :, :]
                self.input_batch = tf.cast(self.input_batch, tf.float32)
                self.input_batch.set_shape([None, config.seq_len, config.frames_per_step, config.out_H, config.out_W, config.input_channels])
                self.batch_size = tf.shape(self.input_batch)[0]
                self.h_input = Input_manager.h_input[device_j, :, :, :]
                self.h_input.set_shape([len(config.encoder_lstm_layers), None, config.lstm_units])
                self.c_input = Input_manager.c_input[device_j, :, :, :]
                self.c_input.set_shape([len(config.encoder_lstm_layers), None, config.lstm_units])
                drop_out_prob = Input_manager.drop_out_prob
                in_lstm_drop_out_prob = Input_manager.in_lstm_drop_out_prob
                out_lstm_drop_out_prob = Input_manager.out_lstm_drop_out_prob
                state_lstm_drop_out_prob = Input_manager.state_lstm_drop_out_prob

            with tf.name_scope('Object_input'):
                self.obj_input = Input_manager.obj_input[device_j, ...]
                flat_obj = tf.contrib.layers.flatten(self.obj_input)

            with tf.name_scope("Now_Target"):
                self.labels = Input_manager.labels[device_j, :, :]
                now_dec_input = tf.concat([tf.fill([self.batch_size, 1], IO_tool.dataset.word_to_id['go']), self.labels], 1)
                self.now_one_hot_label= tf.one_hot(self.labels, depth = self.out_vocab_size)
                now_dec_embed_input = tf.nn.embedding_lookup(Input_manager.dec_embeddings, now_dec_input)
                now_target_len = tf.ones(shape=(self.batch_size), dtype=tf.int32)*(config.seq_len + 1)
            
            with tf.name_scope("Help_Target"):
                self.help_labels = Input_manager.help_labels[device_j, :, :]
                self.help_labels.set_shape([None, 4])
                help_dec_input = tf.concat([tf.fill([self.batch_size, 1], IO_tool.dataset.word_to_id['go']), self.help_labels], 1)
                self.help_one_hot_label= tf.one_hot(self.help_labels, depth = self.out_vocab_size)
                help_dec_embed_input = tf.nn.embedding_lookup(Input_manager.dec_embeddings, help_dec_input)
                help_target_len = tf.ones(shape=(self.batch_size), dtype=tf.int32)*(4)

            with tf.name_scope("Next_Target"):
                self.next_labels = Input_manager.next_labels[device_j, :]
                self.next_one_hot_label= tf.one_hot(self.next_labels, depth = self.out_vocab_size)
                
            def C3d(Tensor):
                with tf.name_scope('C3d'):
                    # Convolution Layer
                    with tf.name_scope("Conv"):
                        conv1 = self.conv3d('conv1', Tensor, wc['wc1'], bc['bc1'])
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
                        conv5 = tf.nn.leaky_relu(conv5, name='relu5a')
                        conv5 = self.conv3d('conv5b', conv5, wc['wc5b'], bc['bc5b'])
                        conv5 = tf.nn.leaky_relu(conv5, name='relu5b')
                        pool5 = self.max_pool('pool5', conv5, k=2)

                    with tf.name_scope('reshape_c3d'):
                        reshape_1_cd = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
                        reshape_2_cd = tf.contrib.layers.flatten(reshape_1_cd)
                    
                    return reshape_2_cd

            with tf.name_scope('c3d_mapfn'):
                reshape_shape = [-1, config.frames_per_step, config.out_H, config.out_W, config.input_channels]
                reshaped_input = tf.reshape(self.input_batch, reshape_shape)
                c3d_out = C3d(reshaped_input)
                self.c3d_out = tf.reshape(c3d_out, [-1, config.seq_len, c3d_out.shape[-1]])

            with tf.name_scope("Dimension_Encoder"):
                dense1_cd = tf.layers.dense(self.c3d_out, config.enc_fc_1)
                dense1_cd = tf.nn.dropout(dense1_cd, drop_out_prob)
                dense2_cd = tf.layers.dense(dense1_cd, config.enc_fc_2)
                dense2_cd = tf.nn.dropout(dense2_cd, drop_out_prob)
                self.out_pL = tf.layers.dense(dense2_cd, config.lstm_units)

            with tf.name_scope("Dimension_Decoder"):
                dense1_cd = tf.layers.dense(self.out_pL, config.enc_fc_2)
                # dense1_cd = tf.nn.dropout(dense1_cd, drop_out_prob)
                dense2_cd = tf.layers.dense(dense1_cd, config.enc_fc_1)
                # dense2_cd = tf.nn.dropout(dense2_cd, drop_out_prob)
                self.autoenc_out = tf.layers.dense(dense2_cd, self.c3d_out.shape[-1])

            with tf.name_scope('insert_obj_for_encoder'):
                encoder_input = self.out_pL
                if config.use_obj:
                    concateneted_encoder_vector = tf.concat([self.out_pL, self.obj_input], -1)
                    encoder_input = tf.layers.dense(concateneted_encoder_vector, config.lstm_units)
                    

            def lstm_cell_with_drop_out():
                lstm_cell = tf.contrib.rnn.LSTMCell(config.lstm_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                lstm_cell =tf.nn.rnn_cell.DropoutWrapper(lstm_cell, 
                                                        input_keep_prob=in_lstm_drop_out_prob, 
                                                        output_keep_prob=out_lstm_drop_out_prob, 
                                                        state_keep_prob=state_lstm_drop_out_prob)
                return lstm_cell
                
            with tf.name_scope("Lstm_encoder"):
                tf.nn.rnn_cell.DropoutWrapper(lstm_cell_with_drop_out(), output_keep_prob=0.5)
                encoder_cells = [lstm_cell_with_drop_out() for n in config.encoder_lstm_layers]
                stacked_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)

                states = []
                for d in range(self.c_input.shape[0]):
                    states.append(tf.contrib.rnn.LSTMStateTuple(self.c_input[d, :, :], self.h_input[d, : , :]))
                states = tuple(states)

                _, encoder_state = tf.nn.dynamic_rnn(stacked_cell, encoder_input,
                                                                    initial_state=states,
                                                                    dtype=tf.float32)

                c_out = tf.expand_dims(encoder_state[0].c, axis=0)
                h_out = tf.expand_dims(encoder_state[0].h, axis=0)
                for d in range(1, self.c_input.shape[0]):
                    c_exp = tf.expand_dims(encoder_state[d].c, axis=0)
                    h_exp = tf.expand_dims(encoder_state[d].h, axis=0)
                    c_out = tf.concat([c_out, c_exp], axis=0)
                    h_out = tf.concat([h_out, h_exp], axis=0)
                self.c_out = c_out
                self.h_out = h_out
                deploy_c = tf.identity(c_out, name="c_out")
                deploy_h = tf.identity(h_out, name="h_out")
                encoder_state = encoder_state[-1]
   
            def decoder_lstm():
                output_layer = tf.layers.Dense(self.out_vocab_size, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
                decoder_cell = lstm_cell_with_drop_out()
                return decoder_cell, output_layer

            def lstm_classifier(logit):
                training_softmax = tf.nn.softmax(logit, name="softmax_out")
                training_predictions = tf.argmax(input=training_softmax, axis=2, name="argmax")
                training_one_hot_prediction= tf.one_hot(training_predictions, depth = training_softmax.shape[-1], name='one_hot')
                return training_softmax, training_predictions, training_one_hot_prediction

            def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size):
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, tf.fill([batch_size], start_of_sequence_id), end_of_sequence_id)
                decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)
                logit = tf.identity(outputs.rnn_output, 'logits')
                return logit

            with tf.name_scope('Now_Decoder_block'):
                now_decoder, now_output_layer = decoder_lstm()

            with tf.name_scope('Now_Decoder_inference'):
                self.inference_logit = decoding_layer_infer(encoder_state, now_decoder, Input_manager.dec_embeddings, IO_tool.dataset.word_to_id['go'],
                                                        IO_tool.dataset.word_to_id['end'], config.seq_len +1 , self.out_vocab_size, now_output_layer,
                                                        self.batch_size)
                paddings = [[0, 0], [0, (config.seq_len + 1)-tf.shape(self.inference_logit)[1]], [0,0 ]]
                self.inference_logit = tf.pad(self.inference_logit, paddings, 'CONSTANT', constant_values = 1)
                self.inference_softmax, self.inference_predictions, self.inference_one_hot_prediction = lstm_classifier(self.inference_logit) 

            with tf.name_scope('Next_classifier'):
                self.inference_softmax.set_shape([None, (config.seq_len + 1), self.out_vocab_size])
                flat_now = tf.contrib.layers.flatten(self.inference_softmax[:,:-1,:])
                if config.use_obj:
                    flat_input = tf.concat([flat_now, flat_obj], 1)
                else:
                    flat_input = flat_now

                next_input = tf.layers.dense(flat_input, config.lstm_units, activation='tanh')
                next_input = tf.reshape(next_input, [-1, 1, config.lstm_units])
                next_decoder_cell = lstm_cell_with_drop_out()
                _, next_out_state = tf.nn.dynamic_rnn(next_decoder_cell, next_input,
                                                                    initial_state=encoder_state,
                                                                    dtype=tf.float32)
                self.next_logit = tf.layers.dense(next_out_state.c, self.number_of_classes)
                self.next_softmax = tf.nn.softmax(self.next_logit, name='softmax_out')
                self.next_predictions = tf.argmax(input=self.next_softmax, axis=1, name="c3d_prediction")
                self.next_one_hot_prediction= tf.one_hot(self.next_predictions, depth = self.next_softmax.shape[-1])

            with tf.name_scope('Help_input_state'):
                self.inference_softmax.set_shape([None, (config.seq_len + 1), self.out_vocab_size])
                flat_now = tf.contrib.layers.flatten(self.inference_softmax[:,:-1,:])
                if config.use_obj:
                    c_comp = tf.concat([encoder_state.c, flat_now, self.next_softmax, flat_obj], 1)
                    h_comp = tf.concat([encoder_state.h, flat_now, self.next_softmax, flat_obj], 1)
                else:
                    c_comp = tf.concat([encoder_state.c, flat_now, self.next_softmax], 1)
                    h_comp = tf.concat([encoder_state.h, flat_now, self.next_softmax], 1)

                new_c = tf.layers.dense(c_comp, config.lstm_units, activation='tanh')
                new_c = tf.layers.dense(new_c, config.lstm_units, activation='tanh')
                new_h = tf.layers.dense(h_comp, config.lstm_units, activation='tanh')
                new_h = tf.layers.dense(new_h, config.lstm_units, activation='tanh')

                help_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            
            with tf.name_scope('Help_Decoder_block'):
                help_decoder, help_output_layer = decoder_lstm()

            with tf.name_scope('Help_Decoder_inference'):
                self.help_inference_logit = decoding_layer_infer(help_state, help_decoder, Input_manager.dec_embeddings, IO_tool.dataset.word_to_id['go'],
                                                        IO_tool.dataset.word_to_id['end'], 4 , self.out_vocab_size, help_output_layer,
                                                        self.batch_size)
                paddings = [[0, 0], [0, 4-tf.shape(self.help_inference_logit)[1]], [0,0 ]]
                self.help_inference_logit = tf.pad(self.help_inference_logit, paddings, 'CONSTANT', constant_values = 1)
                self.help_inference_softmax, self.help_inference_predictions, self.help_inference_one_hot_prediction = lstm_classifier(self.help_inference_logit)  

            def c3d_classifier_dense(x):
                with tf.variable_scope("c3d_classifier_dense", reuse=tf.AUTO_REUSE):
                    out_cd = tf.layers.dense(x, config.pre_class, name="c3d_dense_1", activation='sigmoid')
                    out_cd = tf.nn.dropout(out_cd, drop_out_prob)
                    out_cd_2 = tf.layers.dense(out_cd, config.pre_class, name="c3d_dense_2", activation='sigmoid')
                    out_cd_2 = tf.nn.dropout(out_cd_2, drop_out_prob)
                    logit = tf.layers.dense(x, self.number_of_classes, name="c3d_dense_3", activation='sigmoid')
                return logit

            with tf.name_scope('c3d_classifier'):
                in_c3d_class = dense2_cd
                reshaped_c3d_out = tf.reshape(in_c3d_class , [-1, in_c3d_class.shape[-1]])
                dense_out = tf.layers.dense(reshaped_c3d_out, config.lstm_units)
                dense_out = tf.nn.dropout(dense_out, drop_out_prob)
                dense_out = tf.layers.dense(dense_out, config.lstm_units)
                dense_out = tf.nn.dropout(dense_out, drop_out_prob)
                dense_out = tf.layers.dense(dense_out, self.number_of_classes)
                # dense_out = c3d_classifier_dense(reshaped_c3d_out)
                self.logit_c3d = tf.reshape(dense_out, [-1,in_c3d_class.shape[-2],dense_out.shape[-1]])
                self.softmax_c3d = tf.nn.softmax(self.logit_c3d)
                self.predictions_c3d = tf.argmax(input=self.softmax_c3d, axis=2, name="c3d_prediction")
                self.c3d_one_hot_prediction= tf.one_hot(self.predictions_c3d, depth = self.softmax_c3d.shape[-1])

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
    def __init__(self,Networks, IO_tool):
        with tf.name_scope('Training_and_Metrics'):
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

            with tf.name_scope('weigth'):
                self.now_weight = tf.placeholder(tf.float32, shape=(None, None, config.seq_len + 1), name="now_label")
                self.next_weight = tf.placeholder(tf.float32, shape=(None, None), name="now_label")
                self.help_weight = tf.placeholder(tf.float32, shape=(None, None, 4), name="now_label")


            with tf.name_scope('Metrics'):
                z = 0
                zero = tf.constant(0, dtype=tf.float32)
                for Net in Networks:
                    last_dim_softmax = len(IO_tool.dataset.word_to_id)
                    if z == 0:
                        c3d_pred_conc = Networks[Net].c3d_one_hot_prediction
                        # now_pred_conc = Networks[Net].now_one_hot_prediction
                        now_inf_pred_conc = Networks[Net].inference_one_hot_prediction
                        next_pred_conc = Networks[Net].next_one_hot_prediction
                        now_label_conc = Networks[Net].now_one_hot_label
                        next_label_conc = Networks[Net].next_one_hot_label
                        # help_pred_conc = Networks[Net].help_one_hot_prediction
                        help_inf_pred_conc = Networks[Net].help_inference_one_hot_prediction
                        help_label_conc = Networks[Net].help_one_hot_label
                        predictions_c3d_conc = Networks[Net].predictions_c3d
                        predictions_now_conc = Networks[Net].inference_predictions
                        predictions_help_conc = Networks[Net].help_inference_predictions
                        predictions_next_conc = Networks[Net].next_predictions
                        obj_label_conc = tf.where(tf.not_equal(Networks[Net].obj_input, zero))[:,-1]
                        flat_soft = tf.reshape(Networks[Net].softmax_c3d, [-1, last_dim_softmax])
                        flat_soft = tf.concat([flat_soft,tf.reshape(Networks[Net].inference_softmax, [-1, last_dim_softmax])], axis=0)
                        flat_soft = tf.concat([flat_soft,tf.reshape(Networks[Net].next_softmax, [-1, last_dim_softmax])], axis=0)
                        flat_soft = tf.concat([flat_soft,tf.reshape(Networks[Net].help_inference_softmax, [-1, last_dim_softmax])], axis=0)
                        z +=1
                    else:
                        c3d_pred_conc = tf.concat([c3d_pred_conc, Networks[Net].c3d_one_hot_prediction], axis=0)
                        # now_pred_conc = tf.concat([now_pred_conc, Networks[Net].now_one_hot_prediction], axis=0)
                        now_inf_pred_conc = tf.concat([now_inf_pred_conc, Networks[Net].inference_one_hot_prediction], axis=0)
                        next_pred_conc = tf.concat([next_pred_conc, Networks[Net].next_one_hot_prediction], axis=0)
                        now_label_conc = tf.concat([now_label_conc, Networks[Net].now_one_hot_label], axis=0)
                        next_label_conc = tf.concat([next_label_conc, Networks[Net].next_one_hot_label], axis=0)
                        # help_pred_conc = tf.concat([help_pred_conc, Networks[Net].help_one_hot_prediction], axis=0)
                        help_inf_pred_conc = tf.concat([help_inf_pred_conc, Networks[Net].help_inference_one_hot_prediction], axis=0)
                        help_label_conc = tf.concat([help_label_conc, Networks[Net].help_one_hot_label], axis=0)
                        predictions_c3d_conc = tf.concat([predictions_c3d_conc,Networks[Net].predictions_c3d], axis=0)
                        predictions_now_conc = tf.concat([predictions_now_conc,Networks[Net].inference_predictions], axis=0)
                        predictions_help_conc = tf.concat([predictions_help_conc,Networks[Net].help_inference_predictions], axis=0)
                        predictions_next_conc = tf.concat([predictions_next_conc,Networks[Net].next_predictions], axis=0)
                        obj_label_conc = tf.concat([obj_label_conc,tf.where(tf.not_equal(Networks[Net].obj_input, zero))[:,-1]], axis=0)
                        flat_soft = tf.concat([flat_soft,tf.reshape(Networks[Net].softmax_c3d, [-1, last_dim_softmax])], axis=0)
                        flat_soft = tf.concat([flat_soft,tf.reshape(Networks[Net].inference_softmax, [-1, last_dim_softmax])], axis=0)
                        flat_soft = tf.concat([flat_soft,tf.reshape(Networks[Net].next_softmax, [-1, last_dim_softmax])], axis=0)
                        flat_soft = tf.concat([flat_soft,tf.reshape(Networks[Net].help_inference_softmax, [-1, last_dim_softmax])], axis=0)

                help_action_target = help_label_conc[...,0,:]
                help_obj_target = help_label_conc[...,1,:]
                help_loc_target = help_label_conc[...,2,:]
                help_action_pred = help_inf_pred_conc[...,0,:]
                help_obj_pred = help_inf_pred_conc[...,1,:]
                help_loc_pred = help_inf_pred_conc[...,2,:]
                flat_soft_min = tf.reduce_min(flat_soft, axis=1)
                flat_soft_max = tf.reduce_max(flat_soft, axis=1)
                diff_soft = flat_soft_max -flat_soft_min

                with tf.name_scope('Metrics_calculation'):
                    c3d_precision, c3d_recall, c3d_f1, c3d_accuracy = self.accuracy_metrics(c3d_pred_conc, now_label_conc[:,:-1,:])
                    # now_precision, now_recall, now_f1, now_accuracy = self.accuracy_metrics(now_pred_conc, now_label_conc)
                    inference_precision, inference_recall, inference_f1, inference_accuracy = self.accuracy_metrics(now_inf_pred_conc[:,:-1,:], now_label_conc[:,:-1,:])
                    next_precision, next_recall, next_f1, next_accuracy = self.accuracy_metrics(next_pred_conc, next_label_conc)
                    # help_precision, help_recall, help_f1, help_accuracy = self.accuracy_metrics(help_pred_conc, help_label_conc)
                    help_inference_precision, help_inference_recall, help_inference_f1, inference_accuracy = self.accuracy_metrics(help_inf_pred_conc[:,:-1,:], help_label_conc[:,:-1,:])
                    action_inference_precision, action_inference_recall, action_inference_f1, action_accuracy = self.accuracy_metrics(help_action_pred, help_action_target)
                    object_inference_precision, object_inference_recall, object_inference_f1, object_inference_accuracy = self.accuracy_metrics(help_obj_pred, help_obj_target)
                    place_inference_precision, place_inference_recall, place_inference_f1, place_inference_accuracy = self.accuracy_metrics(help_loc_pred, help_loc_target)
            with tf.name_scope('Loss'):
                c3d_loss_coll = []
                now_loss_coll = []
                help_loss_coll = []
                next_loss_coll = []
                auto_loss_coll = []
                for Net in Networks:
                    with tf.name_scope(Net):
                        with tf.name_scope("C3d_Loss"):
                            cross_entropy_c3d_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].now_one_hot_label[:,:-1,:], logits=Networks[Net].logit_c3d)
                            # c3d_loss = tf.reduce_mean(tf.matmul(self.now_weight[z,:,:-1], cross_entropy_c3d_vec, transpose_b=True))
                            c3d_loss = tf.reduce_sum(cross_entropy_c3d_vec)

                        with tf.name_scope("Now_Loss"):
                            cross_entropy_Now_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].now_one_hot_label[:,:-1,:], logits=Networks[Net].inference_logit[:,:-1,:])
                            # now_loss = tf.reduce_mean(tf.matmul(self.now_weight[z,:,:-1], cross_entropy_Now_vec, transpose_b=True))
                            now_loss = tf.reduce_sum(cross_entropy_Now_vec)

                        with tf.name_scope("help_Loss"):
                            cross_entropy_help_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].help_one_hot_label[:,:-1,:], logits=Networks[Net].help_inference_logit[:,:-1,:])
                            # help_loss = tf.reduce_mean(tf.matmul(self.help_weight[z,:,:-1 ], cross_entropy_help_vec, transpose_b=True))
                            help_loss = tf.reduce_sum(cross_entropy_help_vec)

                        with tf.name_scope("Next_Loss"):
                            cross_entropy_Next_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Networks[Net].next_one_hot_label, logits=Networks[Net].next_logit)
                            # next_loss = tf.reduce_mean(tf.tensordot(self.next_weight[z,...], cross_entropy_Next_vec, axes=1))
                            next_loss = tf.reduce_sum(cross_entropy_Next_vec)

                        with tf.name_scope("Autoencoder_Loss"):
                            auto_enc_loss=tf.reduce_sum(tf.square(Networks[Net].autoenc_out-Networks[Net].c3d_out))

                        c3d_loss_coll.append(c3d_loss)
                        now_loss_coll.append(now_loss)
                        help_loss_coll.append(help_loss)
                        next_loss_coll.append(next_loss)
                        auto_loss_coll.append(auto_enc_loss)

                c3d_loss_sum = sum(c3d_loss_coll)
                now_loss_sum = sum(now_loss_coll)
                next_loss_sum = sum(next_loss_coll)
                auto_enc_loss_sum = sum(auto_loss_coll)
                help_loss_sum = sum(help_loss_coll)

                with tf.name_scope("Global_Loss"):
                    c3d_loss_sum = tf.cast(c3d_loss_sum, tf.float64)
                    now_loss_sum = tf.cast(now_loss_sum, tf.float64)
                    next_loss_sum = tf.cast(next_loss_sum, tf.float64)
                    auto_enc_loss_sum = tf.cast(auto_enc_loss_sum, tf.float64)
                    help_loss_sum = tf.cast(help_loss_sum, tf.float64)
                    c3d_par = tf.clip_by_value(tf.pow(c3d_recall,2), 0, 0.5)
                    now_par = tf.clip_by_value(tf.pow(inference_recall,4), 0, 0.5)
                    next_par = tf.clip_by_value(tf.pow(next_recall,2), 0, 0.5)
                    total_loss = (c3d_par)*(now_par*(next_par*help_loss_sum + (1-next_par)*next_loss_sum) + (1-now_par)*now_loss_sum) + (1 - c3d_par) * c3d_loss_sum + auto_enc_loss_sum
                    
            with tf.name_scope("Optimizer"):
                Train_variable = [v for v in self.variables if 'Openpose' not in v.name.split('/')[0]]
                Train_variable = [v for v in Train_variable if 'MobilenetV1' not in v.name.split('/')[0]]

                starter_learning_rate = config.learning_rate_start
                learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                            10000, 0.9)
                
                self.train_op = tf.contrib.layers.optimize_loss(
                    loss=total_loss,
                    global_step=self.global_step,
                    learning_rate=learning_rate,
                    optimizer='Adam',
                    clip_gradients=config.gradient_clipping_norm,
                    colocate_gradients_with_ops=True,
                    variables=Train_variable)

            with tf.name_scope('Summary'):
                # tf.summary.histogram("c_out", self.c_out)
                # tf.summary.histogram("h_out", self.h_out)
                # tf.summary.histogram("c_in", self.c_input)
                # tf.summary.histogram("h_in", self.h_input)
                # # tf.summary.histogram("labels_target", argmax_labels)
                with tf.name_scope('Loss'):
                    tf.summary.scalar('1_help_Loss', help_loss_sum)
                    tf.summary.scalar('1_Now_Loss', now_loss_sum)
                    tf.summary.scalar('1_Next_Loss', next_loss_sum)
                    tf.summary.scalar('1_C3d_Loss', c3d_loss_sum)
                    tf.summary.scalar('1_learning_rate', learning_rate)

                with tf.name_scope('recall'):
                    tf.summary.scalar('2_c3d_recall', c3d_recall)
                    tf.summary.scalar('2_now_inference_recall', inference_recall)
                    tf.summary.scalar('2_next_recall', next_recall)
                    tf.summary.scalar('2_help_inference_recall', help_inference_recall)
                    # tf.summary.scalar('2_help_train_recall', help_recall)
                    # tf.summary.scalar('2_now_train_recall', now_recall)

                with tf.name_scope('help_recal_by_word'):
                    tf.summary.scalar('3_action_inference_recall', action_inference_recall)
                    tf.summary.scalar('3_object_inference_recall', object_inference_recall)
                    tf.summary.scalar('3_place_inference_recall', place_inference_recall)

                with tf.name_scope('Help'):
                    # tf.summary.histogram("help_action_target", tf.argmax(input=help_action_target, axis=1, name="help_action_target"))
                    # tf.summary.histogram("help_obj_target", tf.argmax(input=help_obj_target, axis=1, name="help_obj_target"))
                    # tf.summary.histogram("help_loc_target", tf.argmax(input=help_loc_target, axis=1, name="help_loc_target"))
                    # tf.summary.histogram("help_action_pred", tf.argmax(input=help_action_pred, axis=1, name="help_action_pred"))
                    # tf.summary.histogram("help_obj_pred", tf.argmax(input=help_obj_pred, axis=1, name="help_obj_pred"))
                    # tf.summary.histogram("help_loc_pred", tf.argmax(input=help_loc_pred, axis=1, name="help_loc_pred"))
                    tf.summary.histogram("Help_classification", predictions_help_conc)
                    tf.summary.histogram("Help_label", tf.argmax(input=help_label_conc, axis=-1))

                with tf.name_scope('Now'):
                    tf.summary.histogram("Now_classification", predictions_now_conc)
                    tf.summary.histogram("Now_label", tf.argmax(input=now_label_conc, axis=-1))

                with tf.name_scope('Next'):
                    tf.summary.histogram("Next_classification", predictions_next_conc)
                    tf.summary.histogram("Next_label", tf.argmax(input=next_label_conc, axis=-1))

                with tf.name_scope('C3d'):
                    tf.summary.histogram("c3d_classification", predictions_c3d_conc)

                tf.summary.histogram("obj_histogram", obj_label_conc)
                tf.summary.histogram("flat_soft", flat_soft)
                tf.summary.histogram("diff_soft", diff_soft)
                tf.summary.scalar('mean_diff_soft', tf.reduce_mean(diff_soft))
                tf.summary.scalar('auto_enc_loss_sum', auto_enc_loss_sum)
                self.merged = tf.summary.merge_all()

                with tf.name_scope('confusion_plot'):
                    self.confusion_image = tf.placeholder(tf.uint8, shape=(None, None, None, None), name="Confusion")
                    self.now_train_confusion = tf.summary.image('now_train', self.confusion_image)
                    self.c3d_train_confusion = tf.summary.image('c3d_train', self.confusion_image)
                    self.next_train_confusion = tf.summary.image('next_train', self.confusion_image)
                    self.help_train_confusion = tf.summary.image('help_train', self.confusion_image)
                    self.now_val_confusion = tf.summary.image('now_val', self.confusion_image)
                    self.c3d_val_confusion = tf.summary.image('c3d_val', self.confusion_image)
                    self.next_val_confusion = tf.summary.image('next_val', self.confusion_image)
                    self.help_val_confusion = tf.summary.image('help_val', self.confusion_image)


            with tf.name_scope('Outputs'):
                self.predictions_now = []
                self.predictions_next = []
                self.predictions_c3d = []
                self.softmax_now = []
                self.softmax_next = []
                self.softmax_help = []
                self.c_out_list = []
                self.h_out_list = []
                for Net in Networks:
                    self.predictions_now.append(Networks[Net].inference_predictions)
                    self.predictions_next.append(Networks[Net].next_predictions)
                    self.predictions_c3d.append(Networks[Net].predictions_c3d)
                    self.softmax_now.append(Networks[Net].inference_softmax)
                    self.softmax_next.append(Networks[Net].next_softmax)
                    self.softmax_help.append(Networks[Net].help_inference_softmax)
                    self.c_out_list.append(Networks[Net].c_out)
                    self.h_out_list.append(Networks[Net].h_out)
                self.predictions_now_conc = predictions_now_conc
                self.predictions_next_conc = predictions_next_conc
                self.predictions_c3d_conc = predictions_c3d_conc
                self.predictions_help_conc = predictions_help_conc

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
