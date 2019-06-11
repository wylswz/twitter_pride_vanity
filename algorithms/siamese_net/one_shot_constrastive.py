from __future__ import division, print_function
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Dense, Lambda, Activation, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import Callback
from keras import backend as K
from sklearn.model_selection import train_test_split
import math
import numpy as np
import random
import pandas as pd
from PIL import Image



def load_data(path):
    dataset = [line.rstrip('\n') for line in open(path)]
    dic_data = {}
    for data in dataset:
        identity, photo = data.split('/')
        if identity not in dic_data.keys():
            dic_data.update({identity: [data]})
        else:
            dic_data[identity].append(data)
    return dic_data


def postive_training_set():
    postive_pair = []
    label = []
    print(list(ptd.keys())[:10])
    for idenity in list(ptd.keys()):
        photo_list = ptd[idenity]
        while len(photo_list) >= 2:
            photo1, photo2 = np.random.choice(photo_list, 2, replace=False)
            postive_pair.append((photo1, photo2))
            label.append(1)
            photo_list.remove(photo1)
            photo_list.remove(photo2)
    return postive_pair, label


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


ptd = load_data('train_list.txt')
ntd = load_data('train_list.txt')

p_train_X, p_train_Y = postive_training_set()
n_train_X, n_train_Y = negative_training_set()
print(len(p_train_X))
print(len(n_train_X))
train_X = p_train_X + n_train_X
train_Y = np.concatenate((p_train_Y, n_train_Y), axis=None)

data = []
for i in range(len(train_X)):
    data.append((train_X[i], train_Y[i]))

random.shuffle(data)


def tranfor_data(data):
    X = []
    for x, y in data:
        x1, x2 = x
        X.append((x1, x2, y))
    return X


data = tranfor_data(data)

def im_decoder(image_filename):
    image = Image.open('train/'+image_filename)
    if image.mode != 'RGB':
            image = image.convert('RGB')
    image = image.resize((128,128))
    image = img_to_array(image)
    image = preprocess_input(image)
    return image

def pair_generator(triples,  datagens, batch_size):
    while True:
        indices = np.random.permutation(np.arange(len(triples)))
        num_batches = len(triples) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            X1 = np.zeros((batch_size, 128, 128, 3))
            X2 = np.zeros((batch_size, 128, 128, 3))
            Y = np.zeros((batch_size, ))
            for i, (image_filename_l, image_filename_r, label) in enumerate(batch):
                if datagens is None or len(datagens) == 0:
                    X1[i] = im_decoder(image_filename_l)
                    X2[i] = im_decoder(image_filename_r)
                else:
                    X1[i] = datagens[0].random_transform(im_decoder(image_filename_l))
                    X2[i] = datagens[1].random_transform(im_decoder(image_filename_r))
                Y[i] = 0 if label == 0 else 1
            yield [X1, X2], Y

data_train, data_val = train_test_split(data, train_size=0.9)
print(len(data_train), len(data_val))

datagen_args = dict(featurewise_center=True,
                    featurewise_std_normalization=True,
                    horizontal_flip=True)

datagens = [ImageDataGenerator(**datagen_args), ImageDataGenerator(**datagen_args)]
train_pair_gen = pair_generator(data_train, datagens, 64)
val_pair_gen = pair_generator(data_val, None, 64)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



def create_based_model(trainable=False):
    pre_trained_model = Xception(weights='imagenet', include_top=False, pooling='avg')
    for layer in pre_trained_model.layers:
        layer.trainable = trainable
    seq = [pre_trained_model]
    if not trainable:
        seq.append(Dense(512))
    return Sequential(seq)


def get_siamese_model():
    input_shape = (128, 128, 3)
    input_tensor_1 = Input(shape=input_shape)
    input_tensor_2 = Input(shape=input_shape)

    based_model = create_based_model(trainable=True)
    vector_1 = based_model(input_tensor_1)
    vector_2 = based_model(input_tensor_2)

    print('vector_1.shape: ', vector_1.shape)
    print('vector_2.shape: ', vector_2.shape)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([vector_1, vector_2])
    print('distance.shape: ', distance)
    model = Model(input=[input_tensor_1, input_tensor_2], output=distance)
    return model

model = get_siamese_model()
model.summary()

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
adm = Adam()
model.compile(loss=contrastive_loss, optimizer=adm, metrics=[accuracy])

"""for siamese net in keras, freezing before saving to prevent loading error occur"""
def freeze(model):
    """Freeze model weights in every layer."""
    for layer in model.layers:
        layer.trainable = False
        if isinstance(layer, Model):
            freeze(layer)


class TrainingHistory(Callback):
    """for saving the history of every steps"""
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('accuracy'))

num_train_steps = math.floor(len(data_train)/64)
num_valid_steps = math.floor(len(data_val)/64)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1)
csv_logger = CSVLogger("model_history_log.csv", append=True)
training_history = TrainingHistory()
model.fit_generator(train_pair_gen,
                              steps_per_epoch=num_train_steps,
                              epochs=32,
                              validation_data = val_pair_gen,
                              validation_steps= num_valid_steps,
                              callbacks=[early_stopping_callback, csv_logger, training_history],
                              workers=3, use_multiprocessing=True)

freeze(model)
model.save('model/xception_contrastive.h5')

df_history = pd.DataFrame(
    {'accs': history.accs,
     'losses': history.losses,
    })
df_history.to_csv('training_history.csv', index=False)