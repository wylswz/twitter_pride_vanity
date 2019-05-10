import tensorflow as tf
import argparse

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.framework import graph_util
slim = tf.contrib.slim

input_checkpoint = '/home/johnny/Documents/TF_CONFIG/finetune/resv2/model.ckpt'
output_file = 'inference_graph.pb'

g = tf.Graph()
with g.as_default():
    image = tf.placeholder(name='input', dtype=tf.float32, shape=[1, 299, 299, 3])
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(image, num_classes=6012, is_training=False)
        predictions = tf.nn.sigmoid(logits, name='multi_predictions')
        saver = tf_saver.Saver()
        input_graph_def = g.as_graph_def()
        sess = tf.Session()
        saver.restore(sess, input_checkpoint)

        output_node_names = "multi_predictions"
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())