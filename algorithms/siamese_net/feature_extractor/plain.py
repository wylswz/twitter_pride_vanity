import tensorflow as tf


def build_net(config):
    features = config["features"]
    scope = config["scope"]
    device_name = "/cpu:0"
    print(tf.test.is_gpu_available)
    if tf.test.is_gpu_available():
        device_name = "/gpu:0"
    with tf.variable_scope(scope):
        with tf.device(device_name):
            input_layer = features

            conv1 = tf.layers.conv2d(
                name="conv1",
                inputs=input_layer,
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                activation=tf.nn.relu,
            )

            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                            pool_size=(2, 2),
                                            strides=2,
                                            )

            conv2 = tf.layers.conv2d(
                name="conv2",
                inputs=pool1,
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                activation=tf.nn.relu
            )

            pool2 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=(2, 2),
                strides=2
            )

            dense = tf.layers.dense(
                name="dense",
                inputs=pool2,
                units=1024,
                activation=tf.nn.relu
            )

    return dense
