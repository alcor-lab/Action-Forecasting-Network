import tensorflo as tf

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

def fc_network(input, number_of_layers, layers_dimension, activation='None', dropout='None')
    if not isinstance(activation, (list,)):
        activation = [activation]
        for i in range(1, number_of_layers)
            activation.append(activation[1])
    else:
        if len(activation) != number_of_layers:
            Error_string = 'Number of activation is different from number of layers/
                           'number of activation given:' + str(len(activation)) +
                           'while number of layer is' + str(number_of_layers)
            raise ValueError(Error_string)

    if not isinstance(activation, (list,)):
        dropout = [dropout]
        for i in range(1, number_of_layers)
            activation.append(dropout[1])
    else
        if len(dropout) != number_of_layers:
            Error_string = 'Number of dropout is different from number of layers/
                           'number of dropout given:' + str(len(dropout)) +
                           'while number of layer is' + str(number_of_layers)
            raise ValueError(Error_string)

    for j in number_of_layers:
        if j = 0:
            fc_layer = tf.layers.dense(input_now, layers_dimension[j], name = 'fc' + str(j), activation=activation[j])
            if dropout[j] != None:
                dropout = tf.nn.dropout(fc_layer, dropout[j])
        else:
            fc_layer = tf.layers.dense(fc_layer, layers_dimension[j], name = 'fc' + str(j, activation=activation[j])
            if dropout[j] != None:
                dropout = tf.nn.dropout(fc_layer, dropout[j])
    return fc_layer
