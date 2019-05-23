# Author: Yenyung Chang <yenyungc@student.unimelb.edu.au>

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import inception_v3
from PIL import Image
import io
import numpy as np
from shared.ModelWrapper import ModelWrapper
from tensorflow import Graph, Session
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras import models
from keras.models import save_model
import matplotlib.pyplot as plt

class FaceVerification(ModelWrapper):
    def __init__(self, model_path):
        super().__init__(model_path=model_path)

    def im_decoder(self, image):
        image = Image.open(io.BytesIO(image))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((128, 128))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = inception_v3.preprocess_input(image)
        return image

    def load(self):
        self.graph1 = Graph()
        with self.graph1.as_default():
            self.session1 = Session()
            with self.session1.as_default():
                self.model = load_model(self.model_path)

    def preprocess_image(self, image_file):
        img = load_img(image_file, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def preprocess(self, input_images):
        image1, image2 = input_images
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)

        return image1, image2

    def findCosineSimilarity(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def findEuclideanDistance(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def verifyFace(self, img1, img2):
        img1_representation = self.model.predict(img1)[0, :]
        img2_representation = self.model.predict(img2)[0, :]

        cosine_similarity = self.findCosineSimilarity(img1_representation, img2_representation)
        euclidean_distance = self.findEuclideanDistance(img1_representation, img2_representation)

        print("Cosine similarity: ", cosine_similarity)
        print("Euclidean distance: ", euclidean_distance)

        return cosine_similarity

    def predict(self, input_images):
        with self.graph1.as_default():
            with self.session1.as_default():
                (image1, image2) = self.preprocess(input_images)
                cos = self.verifyFace(image1, image2)
        print(1-cos)
        return [[1-cos]]


if __name__ == "__main__":
    fv = FaceVerification("/mnt/models/face_comparison/model.h5")
    fv.load()
    image1 = open("1.png", "rb")
    image2 = open("2.png", "rb")
    res = fv.predict([image1, image2])
    print(res)
