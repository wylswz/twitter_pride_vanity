from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input
from keras import backend as K
from PIL import Image
import os
import numpy as np
file_dic={}
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

pos_pair_x = []
pos_pair_y = []
while count < 1000:
    name1 = np.random.choice(key_list, 1, replace = False)
    if len(file_dic[name1[0]]) >= 2:
        photo1, photo2 = np.random.choice(file_dic[name1[0]], 2, replace = False)
        pos_pair_x.append(([photo1, photo2]))
        pos_pair_y.append(1)
        count = count + 1

test_X = neg_pair_x + pos_pair_x
test_Y = neg_pair_y + pos_pair_y


def im_decoder(image):
    image = Image.open(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

model_contrastive = load_model('final_model/xception_contrastive_1.h5', custom_objects={'contrastive_loss': contrastive_loss})
model_absdiff = load_model('final_model/xception_absdiff_0.h5')

###test model
def evaluation(model, mode):
    import time
    start_time = time.time()
    pre_y = []
    if mode == 'contrastive':
        for pair in test_X:
            if model.predict([im_decoder(pair[0]),im_decoder(pair[1])])[0] < 0.5:
                pre_y.append(1)
            else: pre_y.append(0)
    else:
        for pair in test_X:
            if model.predict([im_decoder(pair[0]),im_decoder(pair[1])])[0] > 0.5:
                pre_y.append(1)
            else: pre_y.append(0)
    print("--- %s seconds for ---" % (time.time() - start_time))
    print(mode)
    correct = 0
    for i in range(len(pre_y)):
        if pre_y[i] == test_Y[i]:
            correct = correct + 1
    print("accuracy:", correct / len(pre_y))

    negative = np.array(pre_y[:500])
    fasle_neg = negative.sum()
    true_neg = 500 - fasle_neg
    positive = np.array(pre_y[500:])
    true_pos = positive.sum()
    false_pos = 500 - true_pos
    print("True Positive:", true_pos)
    print("False Positive:", false_pos)
    print("True Negative:", true_neg)
    print("False Negative:", fasle_neg)

evaluation(model_contrastive, 'contrastive')
evaluation(model_absdiff, 'absdiff')