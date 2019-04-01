import tensorflow as tf
import numpy as np
from siamese_net.settings import TRAIN, EVAL
from siamese_net.feature_extractor import plain, resnetv2
from slim.nets import nets_factory
BUILDER_DICT = {
    "plain": plain,
    "resnetv2": resnetv2
}

def load_graph():
    with tf.gfile.GFile(TRAIN.FINE_TUNE_NET_PATH, "rb") as f:
        graph_def:tf.GraphDef = tf.GraphDef()
        file_str = f.read()
        graph_def.ParseFromString(file_str)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="feature_extractor")


    return graph


def build(config, type="plain"):
    return BUILDER_DICT[type].build_net(config)


if __name__ == "__main__":
    image_numpy = np.zeros([256, 256, 3], dtype=np.float)
    config = {
        "features": tf.expand_dims(tf.Variable(image_numpy, dtype=tf.float16), 0),
        "scope": "extractor"
    }

    with tf.Session() as sess:
        extractor = build(config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print(sess.run(extractor))

