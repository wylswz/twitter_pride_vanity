import tensorflow as tf
import tarfile
from PIL import Image
import numpy as np
import os, random
import traceback


# dataset: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

def file_dir_streamer(image_dir):
    mode = "DIFF"
    counter = 0
    counter_threshold = 1000
    file_list = os.walk(image_dir)
    # root, dir, files
    file_dic = {}
    for r in file_list:
        file_dic[r[0]] = r[2]
    same = True
    keys = list(file_dic.keys())
    while True:
        try:
            key = random.choice(keys)
            value = random.choice(file_dic.get(key))
            path = os.path.join(key, value)
            if mode == "DIFF":
                key_ = random.choice(keys)
                if counter >= counter_threshold:
                    mode = "SAME"
            else:
                key_ = key
                if counter >= counter_threshold:
                    mode = "DIFF"


            value_ = random.choice(file_dic.get(key_))
            path_ = os.path.join(key_, value_)
            # print(path, path_)
            if key_ == key:
                label = 0.0
            else:
                label = 1.0
            #print(path, path_)
            counter += 1
            yield path, path_, label
        except Exception as e:
            traceback.print_exc()
            pass



if __name__ == '__main__':
    filename = 'D:\\dev_tools\\dataset\\test.tar.gz'
    path = 'train'
    '''generator = lambda :stream_tar_dataset(filename)
    tf.data.Dataset.from_generator(generator=generator,
                                   output_types=(tf.float32, tf.float32),
                                   output_shapes=((64,), ()))
    print("Start streaming test")'''

    for i in stream_tar_dataset(filename):
        print(i)