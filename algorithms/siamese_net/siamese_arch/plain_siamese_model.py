import tensorflow as tf
from layers.pooling import spatial_pyramid_pooling


class Siamese:
    test_path = None
    train_path = None

    loss = None
    training = None
    train_op = None
    distance = None

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        features = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        features_ = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        labels = tf.placeholder(dtype=tf.float32, shape=[])

        device_name = "/cpu:0"
        print(tf.test.is_gpu_available)
        if tf.test.is_gpu_available():
            device_name = "/gpu:0"
        with tf.device(device_name):
            input_layer = features
            input_layer_ = features_

            conv1 = tf.layers.conv2d(
                name="conv1",
                inputs=input_layer,
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                activation=tf.nn.relu,
            )

            conv1_ = tf.layers.conv2d(
                name="conv1",
                inputs=input_layer_,
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                activation=tf.nn.relu,
                reuse=True
            )

            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                            pool_size=(2, 2),
                                            strides=2,
                                            )
            pool1_ = tf.layers.max_pooling2d(inputs=conv1_,
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

            conv2_ = tf.layers.conv2d(
                name="conv2",
                inputs=pool1_,
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                activation=tf.nn.relu,
                reuse=True
            )

            '''pool2 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=(2, 2),
                strides=2
            )

            pool2_ = tf.layers.max_pooling2d(
                inputs=conv2_,
                pool_size=(2, 2),
                strides=2
            )

            pool2flat = tf.reshape(pool2, (-1, 32 * 32 * 32))
            pool2flat_ = tf.reshape(pool2_, (-1, 32 * 32 * 32))
            dense = tf.layers.dense(
                name="dense",
                inputs=pool2flat,
                units=1024,
                activation=tf.nn.relu
            )
            dense_ = tf.layers.dense(
                name="dense",
                inputs=pool2flat_,
                units=1024,
                activation=tf.nn.relu,
                reuse=True
            )'''
            conv2flat = spatial_pyramid_pooling(inputs=conv2,
                                                bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                name="SPP",
                                                )
            conv2flat_ = spatial_pyramid_pooling(inputs=conv2_,
                                                 bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                 name="SPP",
                                                 reuse=True)

            dense = tf.layers.dense(
                name="dense",
                inputs=conv2flat,
                units=1024,
                activation=tf.nn.relu
            )
            dense_ = tf.layers.dense(
                name="dense",
                inputs=conv2flat_,
                units=1024,
                activation=tf.nn.relu,
                reuse=True
            )
            dropout = tf.layers.dropout(
                inputs=dense,
                rate=0.4,
                training=self.training)

            dropout_ = tf.layers.dropout(
                inputs=dense_,
                rate=0.4,
                training=self.training)
            logits = tf.layers.dense(name="logits", inputs=dropout, units=10)
            logits_ = tf.layers.dense(name="logits", inputs=dropout_, units=10, reuse=True)
            print(input_layer, input_layer_, features)
            distance = tf.norm(logits - logits_)
            loss = (1.0 - labels) * 0.5 * tf.square(tf.norm(logits - logits_)) + \
                   labels * 0.5 * tf.square(tf.maximum(0.0, 1.0 - tf.norm(logits - logits_)))

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

            #train_op = optimizer.minimize(
            #    loss=loss,
            #    global_step=tf.train.get_global_step())
            #self.train_op = train_op

            self.distance = distance

    def train(self):
        self.training = True
        #self.train_op.run(feed_dict={}, session=self.session)

    def predict(self):
        self.training = False
        raise NotImplementedError


if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        snet = Siamese('D:\\dev_tools\\dataset\\train\\', 'D:\\dev_tools\\dataset\\test\\')


