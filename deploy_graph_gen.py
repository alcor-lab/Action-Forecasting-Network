import tensorflow as tf
from tensorflow.python.client import device_lib
import os
from batch_generator_test import IO_manager
from network_seq import activity_network
from network_seq import Training
from network_seq import Input_manager
import pprint
import config
import pprint
pp = pprint.PrettyPrinter(indent=4)


with tf.Session() as sess:
    IO_tool = IO_manager(sess)
    number_of_classes = IO_tool.num_classes
    local_device_protos = device_lib.list_local_devices()
    available_gpus =  [x for x in local_device_protos if x.device_type == 'GPU']
    j=0
    Net_collection = {}
    Input_net = Input_manager(len(available_gpus), IO_tool)
    device = available_gpus[0]
    with tf.variable_scope('Network') as scope:
        Net = activity_network(number_of_classes, Input_net, j, IO_tool)
    IO_tool.start_openPose()

    with tf.name_scope('Saver_and_Loader'):
        with tf.name_scope('whole_saver'):
            whole_saver = tf.train.Saver(name = 'saver')
        
        ckpts = tf.train.latest_checkpoint('./checkpoint')
        vars_in_checkpoint = tf.train.list_variables(ckpts)
        var_rest = []
        for el in vars_in_checkpoint:
            var_rest.append(el[0])
        variables = tf.contrib.slim.get_variables_to_restore()
        var_list = [v for v in variables if v.name.split(':')[0] in var_rest]
        # pp.pprint(var_list)
        loader = tf.train.Saver(var_list=var_list)
        loader.restore(sess, ckpts)
        whole_saver.save(sess, config.deploy_folder)
        deploy_writer = tf.summary.FileWriter("deploy_writer", sess.graph)

