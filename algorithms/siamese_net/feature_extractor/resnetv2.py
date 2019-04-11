import tensorflow as tf
from google.protobuf import text_format
import numpy as np
from siamese_net.model_utils import model_reader

dir(tf.contrib)


def build_net(config, sess):
    tensor_list = model_reader.saved_model_reader(config["saved_model_path"], "text")
    restore_dict = model_reader.get_to_restore_graph(tensor_list, ["resnet_v2_50/ImageInput:0"])
    saver = tf.train.Saver(var_list=restore_dict)
    saver.restore(sess, config['checkpoint_path'])

    return sess.graph


if __name__ == "__main__":
    config = {
        "saved_model_path":"/home/johnny/Desktop/pure_convnet_models/resnet_v2_50/eval.graph",
        "checkpoint_path": '/home/johnny/Documents/TF_CONFIG/finetune/resv2/model.ckpt'
    }
    with tf.Session() as sess:
        feed_image_np = np.zeros([25, 305, 305, 3], dtype=np.float)
        build_net(config,sess)

        input_tensor = sess.graph.get_tensor_by_name("resnet_v2_50/ImageInput:0")
        output_tensor = sess.graph.get_tensor_by_name("resnet_v2_50/conv1/BiasAdd:0")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print(sess.run(output_tensor, feed_dict={
            input_tensor: feed_image_np
        }))

# ERR: Uninitialized variable