import tensorflow as tf
from archive import stream
from archive.siamese_arch.CNN_exercise import cnn_model_fn
from archive.feature_extractor.resnetv2 import build_net

image_dir = '/home/johnny/Documents/TF_CONFIG/dataset/vgg/vggface2_train/train/'
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

    image_resized =tf.image.resize(tf.image.per_image_standardization(image_decoded),size=[128,128])
    image_resized_ =tf.image.resize(tf.image.per_image_standardization(image_decoded_),size=[128,128])
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

    feature_extractor_config = {
        "saved_model_path":"/home/johnny/Desktop/pure_convnet_models/resnet_v2_50/eval.graph",
        "checkpoint_path": '/home/johnny/Documents/TF_CONFIG/finetune/resv2/model.ckpt'
    }

    with tf.Session() as sess:
        graph = build_net(feature_extractor_config, sess)
        input = graph.get_tensor_by_name("resnet_v2_50/ImageInput:0")
        output = graph.get_tensor_by_name("resnet_v2_50/conv1/BiasAdd:0")
    session_config = tf.ConfigProto(log_device_placement=False)
    session_config.gpu_options.per_process_gpu_memory_fraction = 1


    run_config = tf.estimator.RunConfig().replace(session_config=session_config)


    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir='/home/johnny/Documents/TF_CONFIG/model/Siamese/',
        config=run_config,
        params={"input":input}

    )

    estimator.train(
        input_fn=input_func_gen,
        steps=200000000,

    )
