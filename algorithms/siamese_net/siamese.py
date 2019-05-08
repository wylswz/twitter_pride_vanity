from keras.applications import inception_resnet_v2
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input,Dropout, Flatten, Lambda, Activation,AveragePooling1D,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils
from keras.engine import input_layer
from sklearn.model_selection import train_test_split
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import itertools
import os
import time
import random
import keras
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 16} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

def load_data(path):
    dataset = [line.rstrip('\n') for line in open(path)]
    dic_data = {}
    for data in dataset:
        identity,photo = data.split('/')
        if identity not in dic_data.keys():
            dic_data.update({identity:[data]})
        else:
            dic_data[identity].append(data)
    return dic_data

ptd = load_data('/home/johnny/Documents/TF_CONFIG/dataset/vgg/vggface2_train/train/train_list.txt')
ntd = load_data('/home/johnny/Documents/TF_CONFIG/dataset/vgg/vggface2_train/train/train_list.txt')

def postive_training_set():
    postive_pair = []
    label = []
    print(list(ptd.keys())[:10])
    for idenity in list(ptd.keys()):
        photo_list = ptd[idenity]
        while len(photo_list) >=2:
            photo1, photo2 = np.random.choice(photo_list, 2, replace = False)
            postive_pair.append((photo1, photo2))
            label.append(1)
            photo_list.remove(photo1)
            photo_list.remove(photo2)
    return postive_pair, label
p_train_X, p_train_Y = postive_training_set()


def negative_training_set():
    negative_pair = []
    label = []
    id_list = list(ntd.keys())
    while len(id_list) >= 2:
        id1, id2 = np.random.choice(id_list, 2, replace=False)
        id1_photos = ntd[id1]
        id2_photos = ntd[id2]
        while len(id1_photos) >= 1 and len(id2_photos) >= 1:
            photo1 = random.choice(id1_photos)
            photo2 = random.choice(id2_photos)
            negative_pair.append((photo1, photo2))
            label.append(0)
            id1_photos.remove(photo1)
            id2_photos.remove(photo2)
        id_list.remove(id1)
        id_list.remove(id2)

    return negative_pair, label

n_train_X, n_train_Y = negative_training_set()

train_X = p_train_X + n_train_X
train_Y = np.concatenate((p_train_Y, n_train_Y), axis=None)
data = []
for i in range(len(train_X)):
    data.append((train_X[i], train_Y[i]))

random.shuffle(data)

def tranfor_data(data):
    X = []
    for x,y in data:
        x1, x2 = x
        X.append((x1,x2,y))
    return X

data = tranfor_data(data)

IMAGE_DIR = os.path.join('/home/johnny/Documents/TF_CONFIG/dataset/vgg/vggface2_train', "train")


def load_image_cache(image_cache, image_filename):
    image = plt.imread(os.path.join(IMAGE_DIR, image_filename))
    image = imresize(image, (128, 128))
    image = image.astype("float32")
    image = inception_resnet_v2.preprocess_input(image)
    image_cache[image_filename] = image

def im_decoder(image_filename):
    image = plt.imread(os.path.join(IMAGE_DIR, image_filename))
    image = imresize(image, (128, 128))
    image = image.astype("float32")
    image = inception_resnet_v2.preprocess_input(image)
    return image

image_cache = {}
num_pairs = len(data)
#for i, (image_filename_l, image_filename_r, _) in enumerate(data):
#    if i % 1000 == 0:
##        print("images from {:d}/{:d} pairs loaded to cache".format(i, num_pairs))
#    if not image_filename_l in image_cache.keys():
#        load_image_cache(image_cache, image_filename_l)
#    if not image_filename_r in image_cache.keys():
#        load_image_cache(image_cache, image_filename_r)
#3print("images from {:d}/{:d} pairs loaded to cache, COMPLETE".format(i, num_pairs))

def pair_generator(triples, datagens, batch_size=32):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(triples)))
        num_batches = len(triples) // batch_size
        X1 = np.zeros((batch_size, 128, 128, 3))
        X2 = np.zeros((batch_size, 128, 128, 3))
        Y = np.zeros((batch_size, 2))
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            
            for i, (image_filename_l, image_filename_r, label) in enumerate(batch):
                #print(image_filename_l,image_filename_r,label)
                if datagens is None or len(datagens) == 0:
                    X1[i] = im_decoder(image_filename_l)#image_cache[image_filename_l]
                    X2[i] = im_decoder(image_filename_r)#image_cache[image_filename_r]
                else:
                    X1[i] = datagens[0].random_transform(im_decoder(image_filename_l))
                    X2[i] = datagens[1].random_transform(im_decoder(image_filename_r))
                Y[i] = [1, 0] if label == 0 else [0, 1]
            
            yield [X1, X2], Y

def cosine_distance(vecs, normalize=False):
    x, y = vecs
    if normalize:
        x = K.l2_normalize(x, axis=0)
        y = K.l2_normalize(y, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)

def cosine_distance_output_shape(shapes):
    return shapes[0]


def get_siamese_model():
    # create the base pre-trained model
    input_1 = input_layer.Input(shape=((128,128,3)))
    input_2 = input_layer.Input(shape=((128,128,3)))
    base_model_1 = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
    #base_model_2 = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
    ## train previous model or not
    for layer in base_model_1.layers:
        layer.trainable = False
        layer.name = layer.name + "_1"
    #for layer in base_model_2.layers:
    #    layer.name = layer.name + "_2"
    #    layer.trainable = True
    
    #out_conv_1 = base_model_1.get_layer("conv_7b_ac_1").output
    #out_conv_2 = base_model_1.get_layer("conv_7b_ac_1").output
    out_conv_1 = base_model_1(input_1)
    out_conv_2 = base_model_1(input_2)
    #vector_1 = base_model_1.get_layer("avg_pool_1").output
    #vector_2 = base_model_2.get_layer("avg_pool_2").output
    vector_1 = GlobalAveragePooling2D(data_format='channels_last')(out_conv_1)
    vector_1 = Dense(512)(vector_1)
    vector_1 = Dropout(0.2)(vector_1)
    vector_1 = Activation("relu")(vector_1)
    vector_1 = Dense(128)(vector_1)
    vector_1 = Dropout(0.2)(vector_1)
    vector_1 = Activation("relu")(vector_1)
    vector_1 = Dense(32)(vector_1)
    vector_1 = Activation("relu")(vector_1)
    
    vector_2 = GlobalAveragePooling2D(data_format='channels_last')(out_conv_2)
    vector_2 = Dense(512)(vector_2)
    vector_2 = Dropout(0.2)(vector_2)
    vector_2 = Activation("relu")(vector_2)
    vector_2 = Dense(128)(vector_2)
    vector_2 = Dropout(0.2)(vector_2)
    vector_2 = Activation("relu")(vector_2)
    vector_2 = Dense(32)(vector_2)
    vector_2 = Activation("relu")(vector_2)

    # Add a customized layer to compute the absolute difference between the vectors

    distance = Lambda(cosine_distance,
                      output_shape=cosine_distance_output_shape)([vector_1, vector_2])

    #fc1 = Dense(512, kernel_initializer="glorot_uniform")(distance)
    #fc1 = Dropout(0.2)(fc1)
    #fc1 = Activation("relu")(fc1)
    fc1 = Dense(128, kernel_initializer="glorot_uniform")(distance)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Activation("relu")(fc1)
    pred = Dense(2, kernel_initializer="glorot_uniform")(fc1)
    pred = Activation("softmax")(pred)

    siamese_net = Model(inputs=[input_1, input_2], outputs=pred)
    #siamese_net = Model(inputs=[base_model_1.input, base_model_2.input], outputs=pred)
    return siamese_net

model = get_siamese_model()
model.summary()
data_train, data_val = train_test_split(data, train_size=0.8)

datagen_args = dict(
    rescale=1.0/255.0,
    rotation_range=10,
    samplewise_std_normalization=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.2)
datagens = [ImageDataGenerator(**datagen_args),
            ImageDataGenerator(**datagen_args)]
train_pair_gen = pair_generator(data_train, datagens, 16)
val_pair_gen = pair_generator(data_val, None, 16)
adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
num_train_steps = math.floor(len(data_train)/32)
num_valid_steps = math.floor(len(data_val)/32)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50)
checkpoint_callback = ModelCheckpoint('siamese_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit_generator(train_pair_gen,
                              steps_per_epoch=num_train_steps,
                              epochs=32,
                              validation_data = val_pair_gen,
                              validation_steps= num_valid_steps,
                              callbacks=[checkpoint_callback, early_stopping_callback])