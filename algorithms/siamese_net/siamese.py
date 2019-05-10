from __future__ import division, print_function

import os
import random

import math
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications import inception_v3
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import Dense, Input, Lambda, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import pardec

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

def postive_training_set(data):
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

def negative_training_set(data):
    negative_pair=[]
    label = []
    id_list = list(ntd.keys())
    while len(id_list)>= 2:
        id1, id2 = np.random.choice(id_list, 2, replace = False)
        id1_photos = ntd[id1]
        id2_photos = ntd[id2]
        while len(id1_photos) >= 1 and len(id2_photos)>=1:
            photo1 = random.choice(id1_photos)
            photo2 = random.choice(id2_photos)
            negative_pair.append((photo1, photo2))
            label.append(0)
            id1_photos.remove(photo1)
            id2_photos.remove(photo2)
        id_list.remove(id1)
        id_list.remove(id2)

    return negative_pair , label

ptd = load_data('/home/johnny/Documents/TF_CONFIG/dataset/vgg/vggface2_train/train/train_list.txt')
ntd = load_data('/home/johnny/Documents/TF_CONFIG/dataset/vgg/vggface2_train/train/train_list.txt')

p_train_X, p_train_Y = postive_training_set(ptd)
n_train_X, n_train_Y = negative_training_set(ntd)

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

def im_decoder(image_filename):
    image = plt.imread(os.path.join(IMAGE_DIR, image_filename))
    image = imresize(image, (128, 128))
    image = image.astype("float32")
    image = inception_v3.preprocess_input(image)
    return image

'''def pair_generator(triples,  datagens, batch_size=32):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(triples)))
        num_batches = len(triples) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            X1 = np.zeros((batch_size, 128, 128, 3))
            X2 = np.zeros((batch_size, 128, 128, 3))
#             Y = np.zeros((batch_size, 2))
            Y = np.zeros((batch_size, ))
            for i, (image_filename_l, image_filename_r, label) in enumerate(batch):
                if datagens is None or len(datagens) == 0:
                    X1[i] = im_decoder(image_filename_l)
                    X2[i] = im_decoder(image_filename_r)
                else:
                    X1[i] = datagens[0].random_transform(im_decoder(image_filename_l))
                    X2[i] = datagens[1].random_transform(im_decoder(image_filename_r))
#                 Y[i] = [1, 0] if label == 0 else [0, 1]
                Y[i] = 0 if label == 0 else 1
            yield [X1, X2], Y'''


def pair_generator(triples,  datagens, batch_size=32):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(triples)))
        num_batches = len(triples) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            yield (batch, batch_size,)
def decoding_fn(sample):
    (batch, batch_size) = sample
    X1 = np.zeros((batch_size, 128, 128, 3))
    X2 = np.zeros((batch_size, 128, 128, 3))
    #             Y = np.zeros((batch_size, 2))
    Y = np.zeros((batch_size,))
    for i, (image_filename_l, image_filename_r, label) in enumerate(batch):
        if datagens is None or len(datagens) == 0:
            X1[i] = im_decoder(image_filename_l)
            X2[i] = im_decoder(image_filename_r)
        else:
            X1[i] = datagens[0].random_transform(im_decoder(image_filename_l))
            X2[i] = datagens[1].random_transform(im_decoder(image_filename_r))
        #                 Y[i] = [1, 0] if label == 0 else [0, 1]
        Y[i] = 0 if label == 0 else 1
    return [X1, X2], Y


# datagen_args = dict(rotation_range=10,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     zoom_range=0.2)

# datagens = [ImageDataGenerator(), ImageDataGenerator()]
# pair_gen = pair_generator(data, datagens, 32)
# [X1, X2], Y = next(pair_gen)
# X1[0]

data_train, data_val = train_test_split(data, train_size=0.9)
print(len(data_train), len(data_val))

# datagen_args = dict(rescale=1,
#                     samplewise_center=True,
#                     samplewise_std_normalization=True,
#                     rotation_range=10,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     zoom_range=0.2)

datagens = [ImageDataGenerator(), ImageDataGenerator()]
train_pair_gen = pair_generator(data_train, datagens, 32)
train_pair_gen_ = pardec.ParallelDecoder(train_pair_gen, num_workers=6, cache_size=100, decoder=decoding_fn, deque_timeout=1000)
val_pair_gen = pair_generator(data_val, None, 32)
val_pair_gen_ = pardec.ParallelDecoder(val_pair_gen, num_workers=6, cache_size=100, decoder=decoding_fn, deque_timeout=1000)

print(next(train_pair_gen_))
print(next(val_pair_gen_))

from keras.models import Sequential
def create_based_model(trainable=False):
    pre_trained_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
#    pre_trained_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
    for layer in pre_trained_model.layers:
        layer.trainable = trainable
    seq = [pre_trained_model]
    if not trainable:
        seq.append(Dense(512))
    return Sequential(seq)

def get_siamese_model():
    input_tensor_1 = Input(shape=(128, 128, 3))
    input_tensor_2 = Input(shape=(128, 128, 3))

    based_model = create_based_model(trainable=True)
    vector_1 = based_model(input_tensor_1)
    vector_2 = based_model(input_tensor_2)

    print('vector_1.shape: ', vector_1.shape)
    print('vector_2.shape: ', vector_2.shape)

    distance = Lambda(lambda vec: K.abs(vec[0]-vec[1]))([vector_1, vector_2])
    print('distance.shape: ', distance)

    outputs = Dense(1)(distance)
    outputs = Activation("sigmoid")(outputs)


    siamese_net = Model(inputs=[input_tensor_1, input_tensor_2], outputs=outputs)
    return  siamese_net

model = get_siamese_model()

optimizer = Adam(lr=0.0001, decay=5e-4)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

num_train_steps = math.floor(len(data_train)/32)
num_valid_steps = math.floor(len(data_val)/32)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1)
checkpoint_callback = ModelCheckpoint('siamese_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
csv_logger = CSVLogger("model_history_log.csv", append=True)
history = model.fit_generator(train_pair_gen_,
                              steps_per_epoch=num_train_steps,
                              epochs=32,
                              validation_data = val_pair_gen_,
                              validation_steps= num_valid_steps,
                              callbacks=[checkpoint_callback, early_stopping_callback, csv_logger])