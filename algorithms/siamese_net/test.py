import tensorflow as tf
import os
import itertools
from utils import stream
from Nets.CNN_exercise import cnn_model_fn
import ast
import traceback


dataset_path = 'D:\\dev_tools\\dataset\\test.tar.gz'
image_dir = 'D:\\dev_tools\\dataset\\train\\'
generator = lambda: stream.file_dir_streamer(image_dir)


tf.logging.set_verbosity(tf.logging.INFO)

def _parse_function(path, path_, label):
    image_string = tf.read_file(path)
    image_string_ = tf.read_file(path_)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded_ = tf.image.decode_jpeg(image_string_, channels=3)
    #image_decoded = tf.image.rgb_to_grayscale(image_decoded)
    #image_decoded_ = tf.image.rgb_to_grayscale(image_decoded_)
    #tf.image.per_image_standardization(image_decoded)  #


    shape = tf.image.extract_jpeg_shape(image_string)
    shape_ = tf.image.extract_jpeg_shape(image_string_)

    image_resized = tf.image.per_image_standardization(image_decoded)
    image_resized_ = tf.image.per_image_standardization(image_decoded_)
    print(image_resized)
    return {"feature_1": tf.expand_dims(image_resized, 0),
          "feature_2": tf.expand_dims(image_resized_, 0)}, label


def input_func_gen():
    dset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.string, tf.string, tf.float32)
    )

    dset = dset.map(map_func=_parse_function, num_parallel_calls=1)
    #dset = dset.batch(batch_size=100, drop_remainder=True)



    return dset



if __name__ == '__main__':

    session_config = tf.ConfigProto(log_device_placement=False)
    session_config.gpu_options.per_process_gpu_memory_fraction = 1
    run_config = tf.estimator.RunConfig().replace(session_config=session_config)


    
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir='tmp/model',
        config=run_config,

    )

    estimator.train(
        input_fn=input_func_gen,
        steps=200000000,

    )
