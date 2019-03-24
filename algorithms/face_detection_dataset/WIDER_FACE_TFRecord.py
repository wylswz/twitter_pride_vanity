# On branch algorithms

import tensorflow as tf
import re, os, io, codecs, ast
from object_detection.utils import dataset_util
from PIL import  Image
flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

"""
Please properly install tensorflow/models/research/object_detection properly following the instructions
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
before using the code

Setting PATHONPATH is quite important
"""

NUM_WORKERS = 8
CHUNK_SIZE = 1000
from multiprocessing.pool import Pool


ANNOTATION_PATH = "/home/johnny/RCNN/dataset/wider_face_split/wider_face_train_bbx_gt.txt"
TRAIN_IMG_ROOT = "/home/johnny/RCNN/dataset/WIDER_train/images"
TEST_IMG_ROOT = "/home/johnny/RCNN/dataset/WIDER_val/images"
TEST = "/home/johnny/RCNN/dataset/wider_face_split/test"


def record_gen():
    """
    A generator that yields one TF data sample one time
    :return:{"file":"{filepath}", "gt":[ymin, xmin, ymax, xmax]}
    """
    space = r'\s*'
    instance_counter = 0
    with open(ANNOTATION_PATH) as fp:
        while True:
            name = fp.readline()
            if len(name) == 0:
                yield None
                break
            re_match = re.search(r'(\.jpg)', str(name))
            if re_match is not None:
                instance_counter += 1
                num_samples = int(fp.readline())
                data_sample = {"file": str(name).replace('\n', ''), "gt": [None]*num_samples}
                for i in range(num_samples):
                    bbox = fp.readline()
                    bbox = bbox.replace('\n', '')
                    bbox = re.split(' ', bbox)
                    x_min = int(bbox[0])
                    y_min = int(bbox[1])
                    width = int(bbox[2])
                    height = int(bbox[3])
                    x_max = (x_min + width)
                    y_max = y_min + height
                    data_sample["gt"][i] = [y_min, x_min, y_max, x_max]
                print(instance_counter)
                yield data_sample


def tf_sample(sample) -> tf.train.Example:
    """
    Reading data from generator and conver to a tf example
    :param gen: A generator
    :return: tf.train.example
    """
    if sample is None:
        return None
    print(sample)
    filename = sample['file']
    gts = sample['gt']
    filename = os.path.join(TRAIN_IMG_ROOT, filename)
    with open(filename, 'rb') as fp:
        image_fp: Image.Image = Image.open(fp)
        height = image_fp.size[1]
        width = image_fp.size[0]
        filename = filename.encode('utf-8')
        image_format = b'jpeg'
        bio = io.BytesIO()
        image_fp.save(bio, format='jpeg')
        encoded_image_data = bio.getvalue()

    xmins = [b[1]/width for b in gts]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [b[3]/width for b in gts]  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [b[0]/width for b in gts]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [b[2]/width for b in gts]  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [b'human' for _ in gts]  # List of string class name of bounding box (1 per box)
    classes = [1 for _ in gts]  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def schedule(gen):
    try:
        while True:
            param_lsit = [None] * CHUNK_SIZE
            for pc in range(CHUNK_SIZE):
                param_lsit[pc] = next(gen)

            pool = Pool(NUM_WORKERS)
            res = pool.map(tf_sample, param_lsit)
            pool.close()
            pool.join()

            #for p in param_lsit:
            #    tf_sample(p)
    except StopIteration:
        pool = Pool(NUM_WORKERS)
        res = pool.map(tf_sample, param_lsit)
        pool.close()
        pool.join()








if __name__ == "__main__":
    gen = record_gen()
    schedule(gen)
