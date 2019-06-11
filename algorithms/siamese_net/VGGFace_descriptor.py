from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import os
from os.path import isfile, join
import time


def create_model():
    """reconstruct VGG-Very-Deep-16 CNN architecture  """
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model

model = create_model()
#loading the pretrain weight which can download from
#https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view
model.load_weights('model/vgg_face_weights.h5')
model.summary()
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
epsilon = 0.4

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def verifyFace(img1, img2):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0, :]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0, :]
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)

    if (cosine_similarity < epsilon):
        return 1
    else:
        return 0

# load the test image path
for path in os.listdir("/Users/yanyongchang/Desktop/test1/lfw_160"):
    if path == '.DS_Store' or path == 'model' :
            break
    for file in os.listdir("/Users/yanyongchang/Desktop/test1/lfw_160/"+path):
        if file == '.DS_Store':
            break
        if path not in file_dic.keys():
            file_dic[path] = ['lfw_160/'+path+'/'+file]
        else: file_dic[path].append('lfw_160/'+path+'/'+file)

key_list = list(file_dic.keys())
# create 500 negative pairs
neg_pair_x = []
neg_pair_y = []
count = 0
while count < 500:
    name1, name2 = np.random.choice(key_list, 2, replace = False)
    photo1 = np.random.choice(file_dic[name1], 1, replace = False)
    photo2 = np.random.choice(file_dic[name2], 1, replace = False)
    neg_pair_x.append([photo1[0], photo2[0]])
    neg_pair_y.append(0)
    count = count + 1
#create 500 positive pairs
pos_pair_x = []
pos_pair_y = []
while count < 1000:
    name1 = np.random.choice(key_list, 1, replace = False)
    if len(file_dic[name1[0]]) >= 2:
        photo1, photo2 = np.random.choice(file_dic[name1[0]], 2, replace = False)
        pos_pair_x.append(([photo1, photo2]))
        pos_pair_y.append(1)
        count = count + 1
test_X = neg_pair_x+pos_pair_x
test_Y = neg_pair_y+pos_pair_y

#predict the test dataset
start_time = time.time()
pre_y = []
for pair in test_X:
    pre_y.append(verifyFace(pair[0],pair[1]))
print("--- %s seconds ---" % (time.time() - start_time))

correct = 0
for i in range(len(pre_y)):
    if pre_y[i] == test_Y[i]:
        correct = correct + 1
print("accuracy:", correct/len(pre_y))

negative = np.array(pre_y[:500])
fasle_neg = negative.sum()
true_neg = 500-fasle_neg
positive =  np.array(pre_y[500:])
true_pos = positive.sum()
false_pos = 500-true_pos
print("True Positive:", true_pos)
print("False Positive:", false_pos)
print("True Negative:", true_neg)
print("False NegativeL", fasle_neg)
